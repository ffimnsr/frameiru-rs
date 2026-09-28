//! Optional conversion from [`FrameBuffer`] to a Slint image.
//!
//! Only compiled when the `slint` feature is enabled, so headless crates stay
//! free of GUI dependencies.

use crate::buffer::FrameBuffer;
use crate::format::PixelFormat;

use slint::{Image, Rgb8Pixel, SharedPixelBuffer};

/// Converts an RGB8 [`FrameBuffer`] into a Slint [`Image`] with one copy.
///
/// Returns `None` for any non-`Rgb8` format.
pub fn frame_to_slint_image(frame: &FrameBuffer) -> Option<Image> {
    if frame.metadata.format != PixelFormat::Rgb8 {
        return None;
    }
    let pixel_buffer = SharedPixelBuffer::<Rgb8Pixel>::clone_from_slice(
        bytemuck::cast_slice(&frame.data),
        frame.metadata.resolution.width,
        frame.metadata.resolution.height,
    );
    Some(Image::from_rgb8(pixel_buffer))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::FrameBuffer;
    use crate::format::{FrameMetadata, Resolution};

    #[test]
    fn rejects_non_rgb8() {
        let frame = FrameBuffer::new(FrameMetadata {
            sequence: 0,
            timestamp_us: 0,
            resolution: Resolution {
                width: 4,
                height: 4,
            },
            format: PixelFormat::Bgr8,
        });
        assert!(frame_to_slint_image(&frame).is_none());
    }

    #[test]
    fn converts_rgb8() {
        let frame = FrameBuffer::new(FrameMetadata {
            sequence: 0,
            timestamp_us: 0,
            resolution: Resolution {
                width: 4,
                height: 4,
            },
            format: PixelFormat::Rgb8,
        });
        assert!(frame_to_slint_image(&frame).is_some());
    }
}
