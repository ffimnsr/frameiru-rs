//! Looping background video: decodes raw RGB8 frames via the system `ffmpeg`
//! binary on a background thread and publishes the newest frame lock-free.
//!
//! The decoder thread spawns `ffmpeg` piping `rawvideo rgb24` to stdout, paces
//! reads to the video's native frame rate (`ffprobe`), and restarts the
//! process at end-of-stream so the background loops. `VideoSource::current`
//! snapshots the latest frame with `ArcSwap`, so the compositor never blocks
//! on the decoder. Requires `ffmpeg` and `ffprobe` on `PATH`; both are probed
//! at open time so failures surface immediately.

use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use arc_swap::ArcSwap;
use frameiru_core::error::FrameiruError;
use frameiru_core::Resolution;

/// One decoded RGB8 frame (packed, row-major) plus its dimensions.
pub struct VideoFrame {
    pub data: Vec<u8>,
    pub resolution: Resolution,
}

/// Decoded resolution and frame-interval pacing for a video file.
struct Probe {
    width: u32,
    height: u32,
    interval: Duration,
}

/// Live, looping video background source.
pub struct VideoSource {
    path: PathBuf,
    resolution: Resolution,
    current: Arc<ArcSwap<VideoFrame>>,
    stop: Arc<AtomicBool>,
    /// The live `ffmpeg` child the decoder thread is reading (None between
    /// restarts); owned here so `Drop` can kill it without a blocking read.
    child: Arc<Mutex<Option<Child>>>,
    handle: Option<JoinHandle<()>>,
}

impl std::fmt::Debug for VideoSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VideoSource")
            .field("path", &self.path)
            .field("resolution", &self.resolution)
            .finish_non_exhaustive()
    }
}

impl VideoSource {
    /// Probes the video and starts the looping decoder thread.
    ///
    /// Fails fast when `ffmpeg`/`ffprobe` are missing or the file cannot be
    /// probed, so the error surfaces at `update_background` time instead of
    /// mid-stream.
    pub fn open(path: impl Into<PathBuf>) -> Result<Self, FrameiruError> {
        let path = path.into();
        for binary in ["ffmpeg", "ffprobe"] {
            if Command::new(binary).arg("-version").output().is_err() {
                return Err(FrameiruError::Composition(format!(
                    "background video requires `{binary}` on PATH"
                )));
            }
        }
        let probe = probe(&path)?;
        let resolution = Resolution {
            width: probe.width,
            height: probe.height,
        };
        let current = Arc::new(ArcSwap::from_pointee(VideoFrame {
            // All-zero placeholder so `current()` is always Some; the first
            // decoded frame replaces it within one interval.
            data: vec![0; probe.width as usize * probe.height as usize * 3],
            resolution,
        }));
        let stop = Arc::new(AtomicBool::new(false));
        let child = Arc::new(Mutex::new(None));
        let handle = spawn_decoder(
            path.clone(),
            resolution,
            probe.interval,
            Arc::clone(&current),
            Arc::clone(&stop),
            Arc::clone(&child),
        );
        Ok(Self {
            path,
            resolution,
            current,
            stop,
            child,
            handle: Some(handle),
        })
    }

    /// Snapshots the newest decoded frame without blocking the decoder.
    pub fn current(&self) -> arc_swap::Guard<Arc<VideoFrame>> {
        self.current.load()
    }

    pub fn resolution(&self) -> Resolution {
        self.resolution
    }
}

impl Drop for VideoSource {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        // Killing the child closes its stdout pipe, which unblocks the
        // decoder thread's `read_exact` so the join below terminates.
        if let Some(mut child) = self.child.lock().unwrap().take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Decoder loop: spawn `ffmpeg`, read frames paced to the video's rate,
/// respawn at end-of-stream; exit when `stop` is set.
fn spawn_decoder(
    path: PathBuf,
    resolution: Resolution,
    interval: Duration,
    current: Arc<ArcSwap<VideoFrame>>,
    stop: Arc<AtomicBool>,
    child: Arc<Mutex<Option<Child>>>,
) -> JoinHandle<()> {
    thread::Builder::new()
        .name("frameiru-bg-video".into())
        .spawn(move || {
            let frame_size = resolution.area() as usize * 3;
            while !stop.load(Ordering::Relaxed) {
                let mut frame_data = vec![0u8; frame_size];
                let mut next_deadline = Instant::now();
                let Ok(mut proc) = spawn_ffmpeg(&path) else {
                    thread::sleep(Duration::from_secs(1));
                    continue;
                };
                let Some(stdout) = proc.stdout.take() else {
                    let _ = proc.kill();
                    let _ = proc.wait();
                    thread::sleep(Duration::from_secs(1));
                    continue;
                };
                *child.lock().unwrap() = Some(proc);
                let mut out = BufReader::new(stdout);

                let mut ended = false;
                while !stop.load(Ordering::Relaxed) {
                    match out.read_exact(&mut frame_data) {
                        Ok(()) => {
                            current.store(Arc::new(VideoFrame {
                                data: frame_data.clone(),
                                resolution,
                            }));
                            let now = Instant::now();
                            if now < next_deadline {
                                thread::sleep(next_deadline - now);
                            }
                            next_deadline += interval;
                        }
                        // End-of-stream (or ffmpeg error): restart the loop.
                        Err(_) => {
                            ended = true;
                            break;
                        }
                    }
                }

                if let Some(mut proc) = child.lock().unwrap().take() {
                    let _ = proc.kill();
                    let _ = proc.wait();
                }
                if stop.load(Ordering::Relaxed) {
                    break;
                }
                if ended {
                    // Small pause so a permanently failing file does not
                    // spin a spawn/read error loop.
                    thread::sleep(Duration::from_millis(200));
                }
            }
        })
        .expect("spawn background video decoder")
}

fn spawn_ffmpeg(path: &Path) -> std::io::Result<Child> {
    Command::new("ffmpeg")
        .args(["-nostdin", "-v", "error", "-i"])
        .arg(path)
        .args(["-f", "rawvideo", "-pix_fmt", "rgb24", "-an", "pipe:1"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
}

/// Probes size and frame rate with `ffprobe`, e.g.
/// `ffprobe -v error -select_streams v:0 -show_entries
/// stream=width,height,r_frame_rate,avg_frame_rate -of csv=p=0:s=x file`
/// -> `640x480x30/1x30/1`.
fn probe(path: &Path) -> Result<Probe, FrameiruError> {
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,avg_frame_rate",
            "-of",
            "csv=p=0:s=x",
        ])
        .arg(path)
        .output()
        .map_err(|e| {
            FrameiruError::Composition(format!("cannot run ffprobe on {}: {e}", path.display()))
        })?;
    if !out.status.success() {
        return Err(FrameiruError::Composition(format!(
            "cannot probe background video {}: {}",
            path.display(),
            String::from_utf8_lossy(&out.stderr).trim()
        )));
    }
    let line = String::from_utf8_lossy(&out.stdout);
    let line = line.lines().next().ok_or_else(|| {
        FrameiruError::Composition(format!(
            "ffprobe returned no video stream for {}",
            path.display()
        ))
    })?;
    let (width, height, (num, den)) = parse_probe_line(line).ok_or_else(|| {
        FrameiruError::Composition(format!(
            "cannot parse ffprobe output {line:?} for {}",
            path.display()
        ))
    })?;
    Ok(Probe {
        width,
        height,
        interval: Duration::from_secs_f64(den as f64 / num as f64),
    })
}

/// Parses `WxHxnum/den[xavg]`, preferring `r_frame_rate` and falling back to
/// the average; `0/0` falls back to 30 fps.
fn parse_probe_line(line: &str) -> Option<(u32, u32, (u32, u32))> {
    let mut parts = line.trim().split('x');
    let width = parts.next()?.parse().ok()?;
    let height = parts.next()?.parse().ok()?;
    let rate = parse_rate(parts.next()?).or_else(|| parse_rate(parts.next()?));
    Some((width, height, rate.unwrap_or((30, 1))))
}

/// Parses `num/den`; rejects zero denominators and zero rates.
fn parse_rate(text: &str) -> Option<(u32, u32)> {
    let mut parts = text.split('/');
    let num = parts.next()?.parse().ok()?;
    let den = parts.next()?.parse().ok()?;
    (num > 0 && den > 0).then_some((num, den))
}

#[cfg(test)]
mod tests {
    use super::*;

    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    #[test]
    fn parses_rate_tokens() {
        assert_eq!(parse_rate("30/1"), Some((30, 1)));
        assert_eq!(parse_rate("30000/1001"), Some((30000, 1001)));
        assert_eq!(parse_rate("0/0"), None);
        assert_eq!(parse_rate("0/30"), None);
        assert_eq!(parse_rate("30/0"), None);
        assert_eq!(parse_rate("banana"), None);
        assert_eq!(parse_rate("30"), None);
    }

    #[test]
    fn parses_probe_line_with_avg_fallback() {
        assert_eq!(
            parse_probe_line("640x480x30/1x30/1"),
            Some((640, 480, (30, 1)))
        );
        assert_eq!(
            parse_probe_line("1920x1080x30000/1001x30/1"),
            Some((1920, 1080, (30000, 1001)))
        );
        // r_frame_rate unknown -> use avg_frame_rate.
        assert_eq!(
            parse_probe_line("1280x720x0/0x25/1"),
            Some((1280, 720, (25, 1)))
        );
        // Both unknown -> 30 fps.
        assert_eq!(
            parse_probe_line("1280x720x0/0x0/0"),
            Some((1280, 720, (30, 1)))
        );
    }

    #[test]
    fn rejects_garbage_probe_lines() {
        assert_eq!(parse_probe_line(""), None);
        assert_eq!(parse_probe_line("banana"), None);
        assert_eq!(parse_probe_line("640xbananax30/1"), None);
        assert_eq!(parse_probe_line("640x480"), None);
    }

    #[test]
    fn interval_matches_frame_rate() {
        // Duration stores nanoseconds, so tolerance must exceed one tick.
        let probe_30 = Duration::from_secs_f64(1.0 / 30.0);
        let probe_ntsc = Duration::from_secs_f64(1001.0 / 30000.0);
        assert!((probe_30.as_secs_f64() - 1.0 / 30.0).abs() < 1e-6);
        assert!((probe_ntsc.as_secs_f64() - 1001.0 / 30000.0).abs() < 1e-6);
    }

    /// End-to-end decode test, skipped when ffmpeg/ffprobe are unavailable
    /// (e.g. minimal CI images).
    #[test]
    fn decodes_real_video_frames_until_stopped() {
        if Command::new("ffmpeg").arg("-version").output().is_err()
            || Command::new("ffprobe").arg("-version").output().is_err()
        {
            eprintln!("skipping: ffmpeg/ffprobe not on PATH");
            return;
        }
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "frameiru-bg-video-{}-{}.mp4",
            std::process::id(),
            COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
        ));
        let gen = Command::new("ffmpeg")
            .args([
                "-y",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=red:s=64x48:r=10:d=2",
                "-pix_fmt",
                "yuv420p",
            ])
            .arg(&path)
            .output()
            .expect("run ffmpeg generator");
        assert!(
            gen.status.success(),
            "ffmpeg generation failed: {}",
            String::from_utf8_lossy(&gen.stderr)
        );

        let source = VideoSource::open(&path).unwrap();
        assert_eq!(
            source.resolution(),
            Resolution {
                width: 64,
                height: 48
            }
        );

        // Bounded wait for the first non-placeholder (nonzero) frame.
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let frame = source.current();
            if frame.data.iter().any(|&b| b != 0) {
                assert_eq!(frame.data.len(), 64 * 48 * 3);
                assert_eq!(
                    frame.resolution,
                    Resolution {
                        width: 64,
                        height: 48
                    }
                );
                break;
            }
            assert!(
                Instant::now() < deadline,
                "no decoded frame arrived within 5s"
            );
            thread::sleep(Duration::from_millis(50));
        }
        drop(source);
        let _ = std::fs::remove_file(&path);
    }
}
