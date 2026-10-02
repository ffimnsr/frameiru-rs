//! Shared CLI types, argument parsing, and helpers.

pub mod bench;
pub mod control;
pub mod devices;
pub mod run;

pub use bench::BenchArgs;

use std::path::PathBuf;

use anyhow::{bail, Context as _};
use clap::Args;
use frameiru_core::format::Resolution;
use frameiru_core::BackgroundMode;

/// Default control socket: `$XDG_RUNTIME_DIR/frameiru.sock`, falling back to
/// `/tmp/frameiru-<uid>.sock`.
pub fn default_socket() -> PathBuf {
    if let Ok(dir) = std::env::var("XDG_RUNTIME_DIR") {
        if !dir.is_empty() {
            return PathBuf::from(dir).join("frameiru.sock");
        }
    }
    // Cheap uid approximation without a libc dependency: the top 16 bits of
    // the pid correlate with the uid on typical Linux setups only loosely,
    // so prefer the pid itself for uniqueness instead.
    PathBuf::from(format!("/tmp/frameiru-{}.sock", std::process::id()))
}

/// Arguments shared by `run` and `start`.
#[derive(Debug, Clone, Args)]
pub struct RunArgs {
    /// Capture device path.
    #[arg(long, default_value = "/dev/video0")]
    pub device: PathBuf,
    /// Loopback output device path.
    #[arg(long, default_value = "/dev/video10")]
    pub output: PathBuf,
    /// Use a synthetic test-pattern source instead of a camera.
    #[arg(long)]
    pub mock: bool,
    /// Discard composited frames instead of writing to a loopback device.
    #[arg(long)]
    pub null_sink: bool,
    /// ONNX model path; overrides the embedded default. Best used with
    /// `--input-size` + `--normalization` matching the graph.
    #[arg(long)]
    pub model: Option<PathBuf>,
    /// ONNX model input canvas (WxH) when `--model` is given: 320x320 for
    /// u2net/silueta, 1024x1024 for isnet/BiRefNet/rmbg-2.0, 256x256 for
    /// RVM (the embedded default).
    #[arg(long, default_value = "256x256")]
    pub input_size: String,
    /// Tensor normalization for `--model`: `imagenet` (mean/std, u2net/
    /// silueta/isnet/RVM) or `unit` (plain v/255, MediaPipe-style). The
    /// embedded model uses `imagenet`.
    #[arg(long, default_value = "imagenet")]
    pub normalization: String,
    /// Mask EMA blending factor: `off`, or 0.0 (freeze) ..= 1.0 (no
    /// smoothing). Default `off` for instant motion tracking without trailing blur delay.
    #[arg(long, default_value = "off")]
    pub mask_alpha: String,
    /// Dynamic crop & track (ROI zoom) around the subject. Off by default
    /// because cropping to a bounding box causes delay and boundary clipping
    /// during rapid movement.
    #[arg(long)]
    pub roi_zoom: bool,
    /// Cap on segmentation rate; static scenes skip inference entirely
    /// (motion overrides the cap at half the interval). 0 = every frame.
    #[arg(long, default_value_t = 30)]
    pub infer_fps: u32,
    /// Disable guided-filter mask refinement (edge-aware upsampling against
    /// the full-res frame). On by default; fast-GF costs ~2-3 ms at 640x480.
    #[arg(long)]
    pub no_refine_mask: bool,
    /// ORT intra-op threads. Default pins to the physical core count; use
    /// `--threads 2..4` on constrained CPUs. Must be >= 1.
    #[arg(long, value_name = "N")]
    pub threads: Option<usize>,
    /// Foreground mask dilation in pixels (0 disables). Default 0 prevents
    /// mask bulge into nearby background objects.
    #[arg(long, default_value_t = 0)]
    pub mask_dilate: u32,
    /// Soft mask threshold center (0 = off, 1 = max). Default 0.4 collapses
    /// ghosting mid-alphas toward background/subject. `off` disables.
    #[arg(long, default_value = "0.4")]
    pub mask_contrast: String,
    /// Subject fill light: lifts the masked subject toward white
    /// (`0..1`, `0` off). Makes the person look lit without touching the
    /// background.
    #[arg(long, default_value = "0")]
    pub subject_light: String,
    /// Capture width.
    #[arg(long, default_value_t = 640)]
    pub width: u32,
    /// Capture height.
    #[arg(long, default_value_t = 480)]
    pub height: u32,
    /// Background: `passthrough`, `blur:<radius>`, `color:<r,g,b>`, or
    /// `image:<path>`.
    #[arg(long, default_value = "passthrough")]
    pub background: String,
    /// Cap on composited frames per second (0 = uncapped).
    #[arg(long, default_value_t = 30)]
    pub max_fps: u32,
    /// Control socket path; starts the IPC server when set.
    #[arg(long)]
    pub socket: Option<PathBuf>,
}

impl RunArgs {
    pub fn resolution(&self) -> Resolution {
        Resolution {
            width: self.width,
            height: self.height,
        }
    }
}

/// Control-socket selector shared by `stop`/`status`/`set-bg`.
#[derive(Debug, Clone, Args)]
pub struct SocketArg {
    /// Control socket path.
    #[arg(long, default_value = "")]
    pub socket: String,
}

impl SocketArg {
    pub fn socket(&self) -> PathBuf {
        if self.socket.is_empty() {
            default_socket()
        } else {
            PathBuf::from(&self.socket)
        }
    }
}

#[derive(Debug, Clone, Args)]
pub struct SetBgArgs {
    /// Background: `passthrough`, `blur:<radius>`, `color:<r,g,b>`,
    /// `image:<path>`.
    pub background: String,
    /// Control socket path.
    #[arg(long, default_value = "")]
    pub socket: String,
}

impl SetBgArgs {
    pub fn socket(&self) -> PathBuf {
        if self.socket.is_empty() {
            default_socket()
        } else {
            PathBuf::from(&self.socket)
        }
    }
}

#[derive(Debug, Clone, Args)]
pub struct DeviceArg {
    /// Device path, e.g. /dev/video0.
    pub device: PathBuf,
}

/// Parses a mask-smoothing alpha: `off` or a value in [0, 1].
pub fn parse_mask_alpha(s: &str) -> anyhow::Result<Option<f32>> {
    if s.trim().eq_ignore_ascii_case("off") {
        return Ok(None);
    }
    let alpha: f32 = s
        .trim()
        .parse()
        .with_context(|| format!("bad mask-alpha {s:?}"))?;
    if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
        bail!("mask-alpha must be in [0, 1] or `off`, got {s:?}");
    }
    Ok(Some(alpha))
}

/// Parses the soft mask threshold: `off` or a value in [0, 1].
/// `0.0` disables it (no contrast stretch of the mask).
#[allow(dead_code)]
pub fn parse_mask_contrast(s: &str) -> anyhow::Result<f32> {
    if s.trim().eq_ignore_ascii_case("off") {
        return Ok(0.0);
    }
    let c: f32 = s
        .trim()
        .parse()
        .with_context(|| format!("bad mask-contrast {s:?}"))?;
    if !c.is_finite() || !(0.0..=1.0).contains(&c) {
        bail!("mask-contrast must be in [0, 1] or `off`, got {s:?}");
    }
    Ok(c)
}

/// Parses the subject fill light: `off` or a value in [0, 1] (`0` = off).
pub fn parse_subject_light(s: &str) -> anyhow::Result<f32> {
    if s.trim().eq_ignore_ascii_case("off") {
        return Ok(0.0);
    }
    let v: f32 = s
        .trim()
        .parse()
        .with_context(|| format!("bad subject-light {s:?}"))?;
    if !v.is_finite() || !(0.0..=1.0).contains(&v) {
        bail!("subject-light must be in [0, 1] or `off`, got {s:?}");
    }
    Ok(v)
}

/// Parses a normalization scheme for ONNX preprocessing.
#[cfg(feature = "onnx")]
pub fn parse_normalization(s: &str) -> anyhow::Result<frameiru_segment::Normalization> {
    Ok(match s.trim().to_ascii_lowercase().as_str() {
        "imagenet" => frameiru_segment::Normalization::default(),
        "unit" => frameiru_segment::Normalization::unit(),
        other => bail!("unknown normalization {other:?}; use imagenet (mean/std) or unit (v/255)"),
    })
}

pub fn parse_resolution(s: &str) -> anyhow::Result<Resolution> {
    let (w, h) = s
        .split_once('x')
        .ok_or_else(|| anyhow::anyhow!("expected WxH, got {s:?}"))?;
    let width: u32 = w
        .parse()
        .with_context(|| format!("bad width {w:?} in {s:?}"))?;
    let height: u32 = h
        .parse()
        .with_context(|| format!("bad height {h:?} in {s:?}"))?;
    if width == 0 || height == 0 {
        bail!("resolution dimensions must be non-zero: {s}");
    }
    Ok(Resolution { width, height })
}

/// Parses `r,g,b` into a color mode.
pub fn parse_color(s: &str) -> anyhow::Result<BackgroundMode> {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() != 3 {
        bail!("expected color:<r,g,b>, got {s:?}");
    }
    let mut chan = [0u8; 3];
    for (i, part) in parts.iter().enumerate() {
        chan[i] = part
            .trim()
            .parse()
            .with_context(|| format!("bad channel {part:?} in {s:?}"))?;
    }
    Ok(BackgroundMode::Color {
        r: chan[0],
        g: chan[1],
        b: chan[2],
    })
}

/// Parses a `BackgroundMode` from a user-facing string.
pub fn parse_background(s: &str) -> anyhow::Result<BackgroundMode> {
    if s == "passthrough" {
        return Ok(BackgroundMode::Passthrough);
    }
    if let Some(radius) = s.strip_prefix("blur:") {
        let radius: f32 = radius
            .trim()
            .parse()
            .with_context(|| format!("bad blur radius {radius:?}"))?;
        if !(1.0..=30.0).contains(&radius) {
            bail!("blur radius must be in 1..=30");
        }
        return Ok(BackgroundMode::Blur { radius });
    }
    if let Some(color) = s.strip_prefix("color:") {
        return parse_color(color);
    }
    if let Some(path) = s.strip_prefix("image:") {
        if path.trim().is_empty() {
            bail!("image path must not be empty");
        }
        return Ok(BackgroundMode::Image {
            path: PathBuf::from(path.trim()),
        });
    }
    bail!("unknown background {s:?}; use passthrough | blur:<r> | color:<r,g,b> | image:<path>")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_mask_alpha() {
        assert_eq!(parse_mask_alpha("off").unwrap(), None);
        assert_eq!(parse_mask_alpha("0.5").unwrap(), Some(0.5));
        assert_eq!(parse_mask_alpha("0").unwrap(), Some(0.0));
        assert_eq!(parse_mask_alpha("1").unwrap(), Some(1.0));
        for bad in ["1.5", "-0.1", "nan", ""] {
            assert!(parse_mask_alpha(bad).is_err(), "{bad:?} must fail");
        }
    }

    #[test]
    fn parses_subject_light() {
        assert_eq!(parse_subject_light("off").unwrap(), 0.0);
        assert_eq!(parse_subject_light("0").unwrap(), 0.0);
        assert_eq!(parse_subject_light("0.6").unwrap(), 0.6);
        assert_eq!(parse_subject_light("1").unwrap(), 1.0);
        for bad in ["1.5", "-0.1", "nan", ""] {
            assert!(parse_subject_light(bad).is_err(), "{bad:?} must fail");
        }
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn parses_normalization_schemes() {
        assert_eq!(
            parse_normalization("imagenet").unwrap(),
            frameiru_segment::Normalization::default()
        );
        assert_eq!(
            parse_normalization("unit").unwrap(),
            frameiru_segment::Normalization::unit()
        );
        assert!(parse_normalization("nope").is_err());
    }

    #[test]
    fn parses_resolutions() {
        assert_eq!(
            parse_resolution("640x480").unwrap(),
            Resolution {
                width: 640,
                height: 480
            }
        );
        assert!(parse_resolution("640").is_err());
        assert!(parse_resolution("0x10").is_err());
        assert!(parse_resolution("x10").is_err());
    }

    #[test]
    fn parses_background_modes() {
        assert_eq!(
            parse_background("passthrough").unwrap(),
            BackgroundMode::Passthrough
        );
        assert_eq!(
            parse_background("blur:12.5").unwrap(),
            BackgroundMode::Blur { radius: 12.5 }
        );
        assert_eq!(
            parse_background("color:1, 2, 3").unwrap(),
            BackgroundMode::Color { r: 1, g: 2, b: 3 }
        );
        assert_eq!(
            parse_background("image:/tmp/bg.png").unwrap(),
            BackgroundMode::Image {
                path: "/tmp/bg.png".into()
            }
        );
        for bad in ["blur:0", "blur:31", "color:1,2", "color:1,2,x", "nope", ""] {
            assert!(parse_background(bad).is_err(), "{bad:?} must fail");
        }
    }

    #[test]
    fn default_socket_is_absolute() {
        let path = default_socket();
        assert!(path.is_absolute());
        assert!(path.to_string_lossy().contains("frameiru"));
    }
}
