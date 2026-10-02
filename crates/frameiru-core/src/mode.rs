//! Background replacement modes for the compositor.

use std::path::PathBuf;

/// Fully dynamic background mode. The engine applies it live without
/// restarting the camera stream.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BackgroundMode {
    /// Zero-cost bypass: the source frame passes through untouched.
    Passthrough,
    /// Blur the background with a blur radius in pixels.
    Blur { radius: f32 },
    /// Replace the background with a solid color (RGB).
    Color { r: u8, g: u8, b: u8 },
    /// Replace the background with an image loaded from `path`.
    Image { path: PathBuf },
    /// Replace the background with a looping video decoded from `path`.
    Video { path: PathBuf },
}

/// Full-frame post-processing overlay applied on top of the composite.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum OverlayMode {
    /// No overlay.
    #[default]
    None,
    /// Subtle darkening of alternating rows (classic display scanlines).
    Scanlines,
    /// Warm animated light patch drifting across the frame.
    LightLeak,
    /// CRT look: stronger scanlines, vignette, and a rolling brightness flicker.
    Crt,
}
