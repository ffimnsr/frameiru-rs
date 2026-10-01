//! Resolved background states shared by the CPU and GPU compositors.
//!
//! [`BackgroundMode`](frameiru_core::BackgroundMode) is the dynamic control
//! surface; [`Background`] is the loaded, render-ready version: images are
//! decoded once at update time instead of per frame.

use frameiru_core::error::FrameiruError;
use frameiru_core::{BackgroundMode, Resolution};

/// Render-ready background for one composite session.
#[derive(Debug, Clone)]
pub enum Background {
    /// Output the source frame untouched.
    Passthrough,
    /// Halo-free foreground-aware blur behind the subject (foreground pixels
    /// are excluded from the kernel so subject colors cannot smear the edge).
    Blur { radius: u32 },
    /// Solid color background.
    Color { r: u8, g: u8, b: u8 },
    /// Decoded RGB8 image, stretched to the frame on sampling.
    Image {
        data: Vec<u8>,
        resolution: Resolution,
    },
}

impl Background {
    /// Resolves a control-plane mode into a render-ready background.
    ///
    /// Image paths are decoded immediately, so failures (missing file,
    /// corrupt data) surface at `update_background` time, not mid-stream.
    pub fn resolve(mode: BackgroundMode) -> Result<Self, FrameiruError> {
        match mode {
            BackgroundMode::Passthrough => Ok(Self::Passthrough),
            BackgroundMode::Blur { radius } => {
                let radius = radius.round();
                if !radius.is_finite() || !(1.0..=30.0).contains(&radius) {
                    return Err(FrameiruError::InvalidArgument(format!(
                        "blur radius must be in 1..=30, got {radius}"
                    )));
                }
                Ok(Self::Blur {
                    radius: radius as u32,
                })
            }
            BackgroundMode::Color { r, g, b } => Ok(Self::Color { r, g, b }),
            BackgroundMode::Image { path } => {
                let img = image::open(&path).map_err(|e| {
                    FrameiruError::Composition(format!(
                        "cannot load background image {}: {e}",
                        path.display()
                    ))
                })?;
                let rgb = img.to_rgb8();
                let (w, h) = rgb.dimensions();
                Ok(Self::Image {
                    data: rgb.into_raw(),
                    resolution: Resolution {
                        width: w,
                        height: h,
                    },
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_scalar_modes() {
        assert!(matches!(
            Background::resolve(BackgroundMode::Passthrough).unwrap(),
            Background::Passthrough
        ));
        assert!(matches!(
            Background::resolve(BackgroundMode::Blur { radius: 12.7 }).unwrap(),
            Background::Blur { radius: 13 }
        ));
        assert!(matches!(
            Background::resolve(BackgroundMode::Color { r: 1, g: 2, b: 3 }).unwrap(),
            Background::Color { r: 1, g: 2, b: 3 }
        ));
    }

    #[test]
    fn rejects_out_of_range_blur() {
        assert!(Background::resolve(BackgroundMode::Blur { radius: 0.0 }).is_err());
        assert!(Background::resolve(BackgroundMode::Blur { radius: 31.0 }).is_err());
        assert!(Background::resolve(BackgroundMode::Blur { radius: f32::NAN }).is_err());
    }

    #[test]
    fn missing_image_fails_at_resolve() {
        let err = Background::resolve(BackgroundMode::Image {
            path: "/nonexistent/bg.png".into(),
        });
        assert!(err.is_err());
    }
}
