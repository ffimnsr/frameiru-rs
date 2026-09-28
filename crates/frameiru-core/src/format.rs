//! Pixel formats, resolutions, and frame metadata.

use crate::error::FrameiruError;

/// Raw pixel format carried by a [`FrameBuffer`](crate::FrameBuffer).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PixelFormat {
    /// 24-bit RGB, tightly packed, 3 bytes per pixel.
    Rgb8,
    /// 24-bit BGR, tightly packed, 3 bytes per pixel.
    Bgr8,
    /// 4:2:2 YUV with a 4-byte `Y0 U0 Y1 V0` macro-pixel, 2 bytes per pixel.
    Yuyv422,
    /// 4:2:0 YUV, full-resolution Y plane plus interleaved UV plane,
    /// 3/2 bytes per pixel.
    Nv12,
}

impl PixelFormat {
    /// Number of color channels stored per pixel.
    pub fn channels(&self) -> u32 {
        match self {
            PixelFormat::Rgb8 | PixelFormat::Bgr8 => 3,
            PixelFormat::Yuyv422 | PixelFormat::Nv12 => 2,
        }
    }

    /// Exact number of bits used to store one pixel.
    pub fn bits_per_pixel(&self) -> u32 {
        match self {
            PixelFormat::Rgb8 | PixelFormat::Bgr8 => 24,
            PixelFormat::Yuyv422 => 16,
            PixelFormat::Nv12 => 12,
        }
    }

    /// Total byte size of a frame at `resolution`.
    ///
    /// Note: real drivers may require per-row alignment for `Yuyv422` and
    /// `Nv12`; this helper returns the tightly packed size.
    pub fn bytes_per_frame(&self, resolution: Resolution) -> usize {
        let (w, h) = (resolution.width, resolution.height);
        match self {
            PixelFormat::Rgb8 | PixelFormat::Bgr8 => (w * h * 3) as usize,
            PixelFormat::Yuyv422 => (w * h * 2) as usize,
            PixelFormat::Nv12 => (w * h * 3 / 2) as usize,
        }
    }
}

/// A 2D frame resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Resolution {
    pub width: u32,
    pub height: u32,
}

impl Resolution {
    /// Creates a resolution, rejecting zero dimensions.
    pub fn new(width: u32, height: u32) -> Result<Self, FrameiruError> {
        if width == 0 || height == 0 {
            return Err(FrameiruError::InvalidArgument(format!(
                "resolution dimensions must be non-zero, got {width}x{height}"
            )));
        }
        Ok(Self { width, height })
    }

    /// Whether both dimensions are non-zero.
    pub const fn is_valid(&self) -> bool {
        self.width != 0 && self.height != 0
    }

    /// Total pixel count as `u64` to avoid overflow on large resolutions.
    pub const fn area(&self) -> u64 {
        self.width as u64 * self.height as u64
    }
}

/// Per-frame bookkeeping attached to a [`FrameBuffer`](crate::FrameBuffer).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FrameMetadata {
    /// Monotonic frame sequence number assigned by the producer.
    pub sequence: u64,
    /// Capture timestamp in microseconds (CLOCK_MONOTONIC).
    pub timestamp_us: u64,
    pub resolution: Resolution,
    pub format: PixelFormat,
}

impl FrameMetadata {
    /// Tightly packed byte size of the frame described by this metadata.
    pub fn bytes_len(&self) -> usize {
        self.format.bytes_per_frame(self.resolution)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolution_helpers() {
        let res = Resolution::new(640, 480).unwrap();
        assert_eq!(res.area(), 640 * 480);
        assert!(res.is_valid());
        assert!(Resolution::new(0, 480).is_err());
        assert!(Resolution::new(640, 0).is_err());
        assert!(!Resolution {
            width: 0,
            height: 480
        }
        .is_valid());
    }

    #[test]
    fn tight_frame_sizes() {
        let res = Resolution {
            width: 640,
            height: 480,
        };
        assert_eq!(PixelFormat::Rgb8.bytes_per_frame(res), 921_600);
        assert_eq!(PixelFormat::Bgr8.bytes_per_frame(res), 921_600);
        assert_eq!(PixelFormat::Yuyv422.bytes_per_frame(res), 614_400);
        assert_eq!(PixelFormat::Nv12.bytes_per_frame(res), 460_800);
    }

    #[test]
    fn bits_per_pixel() {
        assert_eq!(PixelFormat::Rgb8.bits_per_pixel(), 24);
        assert_eq!(PixelFormat::Bgr8.bits_per_pixel(), 24);
        assert_eq!(PixelFormat::Yuyv422.bits_per_pixel(), 16);
        assert_eq!(PixelFormat::Nv12.bits_per_pixel(), 12);
    }

    #[test]
    fn metadata_bytes_len() {
        let meta = FrameMetadata {
            sequence: 1,
            timestamp_us: 1234,
            resolution: Resolution {
                width: 640,
                height: 480,
            },
            format: PixelFormat::Rgb8,
        };
        assert_eq!(meta.bytes_len(), 921_600);
    }
}
