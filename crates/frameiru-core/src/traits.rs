//! Engine boundary traits implemented by capture, segment, compose, and sink.

use crate::buffer::{FrameBuffer, Mask};
use crate::error::FrameiruError;
use crate::format::{PixelFormat, Resolution};
use crate::mode::BackgroundMode;

/// Produces video frames (V4L2 camera, mock test pattern, etc.).
pub trait FrameSource: Send + 'static {
    /// Fixed resolution of the frames this source emits.
    fn resolution(&self) -> Resolution;

    /// Pixel format of the frames this source emits.
    fn format(&self) -> PixelFormat;

    /// Blocks until the next frame is available.
    fn next_frame(&mut self) -> Result<FrameBuffer, FrameiruError>;
}

/// Produces a foreground [`Mask`] for a source frame (ONNX model, etc.).
pub trait Segmenter: Send + 'static {
    /// Resolution the segmenter expects as input.
    fn input_resolution(&self) -> Resolution;

    /// Segments `frame` into a normalized soft mask.
    fn segment(&mut self, frame: &FrameBuffer) -> Result<Mask, FrameiruError>;

    /// Clears any internal temporal state (recurrent models such as RVM).
    /// Stateless models may ignore this; the default is a no-op. The
    /// pipeline calls this when the capture stream drops frames so recurrent
    /// state cannot ghost across discontinuities.
    fn reset_state(&mut self) {}
}

/// Blends a source frame with a mask and a background mode.
pub trait Compositor: Send + 'static {
    /// Composites `source` + `mask` into `output`.
    fn composite(
        &mut self,
        source: &FrameBuffer,
        mask: &Mask,
        output: &mut FrameBuffer,
    ) -> Result<(), FrameiruError>;

    /// Applies a new background mode at runtime.
    fn update_background(&mut self, mode: BackgroundMode) -> Result<(), FrameiruError>;

    /// Subject fill light `0..=1`: the composite is lifted toward white by
    /// `light * mask * (1 - out)` — brightens only the foreground ("lit").
    /// `0.0` disables. Default no-op for custom compositors.
    fn set_subject_light(&mut self, _light: f32) {}
}

/// Consumes composited frames (v4l2loopback writer, preview broadcast, etc.).
pub trait FrameSink: Send + 'static {
    /// Writes a frame to the sink.
    fn write_frame(&mut self, frame: &FrameBuffer) -> Result<(), FrameiruError>;
}
