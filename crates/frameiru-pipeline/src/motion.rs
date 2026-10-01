//! Motion gating: cheap luma signatures and an inference cadence gate.
//!
//! Webcam scenes are mostly static, so segmenting every captured frame is
//! waste: the mask (and its EMA) only needs refreshing when the scene
//! actually changes. [`InferGate`] throttles inference to `infer_max_fps`
//! and only breaks the cap when its luma signature says the scene moved.

use std::time::{Duration, Instant};

use frameiru_core::error::FrameiruError;
use frameiru_core::FrameBuffer;

/// Horizontal signature cells; the grid is capped to the frame size, so
/// tiny test frames degrade gracefully.
const SIG_W: u32 = 32;
/// Vertical signature cells.
const SIG_H: u32 = 24;
/// Sample every 2nd pixel per cell when accumulating luma.
const SAMPLE_STEP: u32 = 2;

/// BT.601 luma (scaled to 8-bit) averaged per cell of a `SIG_W x SIG_H`
/// grid, sampled at `SAMPLE_STEP` for cost. Deterministic and cheap:
/// ~0.03 ms at 640x480.
pub fn luma_signature(frame: &FrameBuffer) -> Result<Vec<u8>, FrameiruError> {
    let (w, h) = (
        frame.metadata.resolution.width,
        frame.metadata.resolution.height,
    );
    if w == 0 || h == 0 {
        return Err(FrameiruError::InvalidArgument(
            "signature requires a non-zero frame".into(),
        ));
    }
    if frame.data.len() < (w * h * 3) as usize {
        return Err(FrameiruError::InsufficientCapacity {
            need: (w * h * 3) as usize,
            have: frame.data.len(),
        });
    }
    let cw = SIG_W.min(w);
    let ch = SIG_H.min(h);
    let mut out = Vec::with_capacity((cw * ch) as usize);
    for cy in 0..ch {
        let y0 = (cy * h / ch) as usize;
        let y1 = (((cy + 1) * h / ch) as usize).max(y0 + 1).min(h as usize);
        for cx in 0..cw {
            let x0 = (cx * w / cw) as usize;
            let x1 = (((cx + 1) * w / cw) as usize).max(x0 + 1).min(w as usize);
            let mut acc = 0.0f64;
            let mut n = 0usize;
            let mut y = y0;
            while y < y1 {
                let base = y * w as usize;
                let mut x = x0;
                while x < x1 {
                    let i = (base + x) * 3;
                    acc += 0.299 * frame.data[i] as f64
                        + 0.587 * frame.data[i + 1] as f64
                        + 0.114 * frame.data[i + 2] as f64;
                    n += 1;
                    x += SAMPLE_STEP as usize;
                }
                y += SAMPLE_STEP as usize;
            }
            let v = (acc / n.max(1) as f64).round();
            out.push(v as u8);
        }
    }
    Ok(out)
}

/// Mean absolute luma difference between two signatures, in 8-bit units
/// (0 = identical, ~255 = maximal).
pub fn luma_diff(a: &[u8], b: &[u8]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let sum: u64 = a[..n]
        .iter()
        .zip(&b[..n])
        .map(|(x, y)| (*x as i32 - *y as i32).unsigned_abs() as u64)
        .sum();
    sum as f32 / n as f32
}

/// Throttles segmentation to a cadence, with a motion override.
///
/// - Frames spaced >= `interval` apart are always segmented.
/// - A frame with above-threshold motion is segmented early, but never
///   sooner than `min_interval` (rate floor).
/// - Static frames between the caps are skipped; the EMA mask carries over.
pub struct InferGate {
    interval: Duration,
    min_interval: Duration,
    motion_threshold: f32,
    last: Option<Instant>,
    last_sig: Vec<u8>,
}

impl InferGate {
    /// `infer_max_fps == 0` disables throttling (every frame is segmented).
    pub fn new(infer_max_fps: u32, motion_threshold: f32) -> Self {
        let interval = if infer_max_fps == 0 {
            Duration::ZERO
        } else {
            Duration::from_secs_f64(1.0 / infer_max_fps as f64)
        };
        // Motion override floor: never burst above ~2x the cap.
        let min_interval = if interval.is_zero() {
            Duration::ZERO
        } else {
            interval / 2
        };
        Self {
            interval,
            min_interval,
            motion_threshold,
            last: None,
            last_sig: Vec::new(),
        }
    }

    /// Whether `frame` should be segmented now. Always true when throttling
    /// is disabled or on the first frame.
    pub fn should_infer(&mut self, frame: &FrameBuffer, now: Instant) -> bool {
        if self.interval.is_zero() {
            return true;
        }
        let sig = match luma_signature(frame) {
            Ok(s) => s,
            Err(_) => return true, // cannot measure: don't gate away the mask
        };
        let motion = match self.last {
            Some(_) => luma_diff(&sig, &self.last_sig),
            None => f32::MAX,
        };
        self.last_sig = sig;
        let Some(t0) = self.last else {
            self.last = Some(now);
            return true;
        };
        let elapsed = now.saturating_duration_since(t0);
        let urgent = motion >= self.motion_threshold && elapsed >= self.min_interval;
        if elapsed >= self.interval || urgent {
            self.last = Some(now);
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use frameiru_core::format::{FrameMetadata, PixelFormat};

    fn frame(w: u32, h: u32, fill: u8) -> FrameBuffer {
        let mut f = FrameBuffer::new(FrameMetadata {
            sequence: 0,
            timestamp_us: 0,
            resolution: Resolution {
                width: w,
                height: h,
            },
            format: PixelFormat::Rgb8,
        });
        f.data = vec![fill; (w * h * 3) as usize];
        f
    }

    use frameiru_core::Resolution;

    fn sig_of(fill: u8) -> Vec<u8> {
        luma_signature(&frame(64, 48, fill)).unwrap()
    }

    #[test]
    fn identical_frames_have_zero_diff() {
        assert_eq!(luma_diff(&sig_of(100), &sig_of(100)), 0.0);
    }

    #[test]
    fn changed_frames_have_large_diff() {
        let d = luma_diff(&sig_of(10), &sig_of(240));
        assert!(d > 100.0, "diff was {d}");
    }

    #[test]
    fn signature_respects_frame_size() {
        // Grid caps to the frame: any non-zero frame yields a signature.
        let sig = sig_of(77);
        assert!(!sig.is_empty());
        assert!(sig.iter().all(|&v| v == 77));
    }

    #[test]
    fn gate_first_frame_infers() {
        let mut g = InferGate::new(10, 2.0);
        assert!(g.should_infer(&frame(32, 24, 50), Instant::now()));
    }

    #[test]
    fn gate_throttles_static_frames() {
        let mut g = InferGate::new(10, 2.0); // 100 ms cap
        let t0 = Instant::now();
        assert!(g.should_infer(&frame(32, 24, 50), t0));
        // Same scene 10 ms later: within the cap, no motion -> skip.
        assert!(!g.should_infer(&frame(32, 24, 50), t0 + Duration::from_millis(10)));
        // Static frame after the cap: due -> infer.
        assert!(g.should_infer(&frame(32, 24, 50), t0 + Duration::from_millis(101)));
    }

    #[test]
    fn gate_motion_overrides_cap() {
        let mut g = InferGate::new(10, 2.0); // cap 100 ms, floor 50 ms
        let t0 = Instant::now();
        assert!(g.should_infer(&frame(32, 24, 50), t0));
        // Big change at 60 ms (past floor, under cap): urgent -> infer.
        assert!(g.should_infer(&frame(32, 24, 200), t0 + Duration::from_millis(60)));
        // Small change at 75 ms after that: no motion, under cap -> skip.
        assert!(!g.should_infer(&frame(32, 24, 210), t0 + Duration::from_millis(75)));
    }

    #[test]
    fn gate_unlimited_infers_every_frame() {
        let mut g = InferGate::new(0, 2.0);
        let t0 = Instant::now();
        assert!(g.should_infer(&frame(32, 24, 50), t0));
        assert!(g.should_infer(&frame(32, 24, 50), t0 + Duration::from_millis(1)));
    }
}
