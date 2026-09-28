//! Synthetic animated test-pattern source.
//!
//! Generates RGB8 frames deterministically from a sequence counter: a moving
//! hue-rotating disc over a vertical gradient with scrolling bars. Powers
//! headless CI tests and GUI development without a physical camera.

use frameiru_core::error::FrameiruError;
use frameiru_core::format::{FrameMetadata, PixelFormat, Resolution};
use frameiru_core::traits::FrameSource;
use frameiru_core::{timestamp_us_now, FrameBuffer};

/// Animated procedural RGB8 frame source.
pub struct MockSource {
    resolution: Resolution,
    sequence: u64,
}

impl MockSource {
    /// Creates a source that emits `resolution`-sized RGB8 frames.
    pub fn new(resolution: Resolution) -> Result<Self, FrameiruError> {
        if !resolution.is_valid() {
            return Err(FrameiruError::InvalidArgument(format!(
                "mock source resolution must be non-zero, got {resolution:?}"
            )));
        }
        Ok(Self {
            resolution,
            sequence: 0,
        })
    }

    /// Sequence number that will be assigned to the next emitted frame.
    pub fn sequence(&self) -> u64 {
        self.sequence
    }

    fn pattern(x: u32, y: u32, w: u32, h: u32, sequence: u64) -> (u8, u8, u8) {
        let wf = w as f32;
        let hf = h as f32;
        // Disc center orbits slowly with the sequence counter.
        let cx = wf * 0.5 + (sequence as f32 * 0.35).sin() * wf * 0.22;
        let cy = hf * 0.5 + (sequence as f32 * 0.22).cos() * hf * 0.22;
        let dx = x as f32 - cx;
        let dy = y as f32 - cy;
        let radius = hf * 0.18;
        let in_disc = dx * dx + dy * dy <= radius * radius;

        let gradient = ((y as f32 / hf * 200.0) as u8) + 22;
        if in_disc {
            hsv_wheel((sequence % 360) as f32 / 360.0 * 6.0)
        } else {
            let shift = (sequence % 32) as u32;
            let band = ((x + shift) / 12 + y / 12).is_multiple_of(2);
            if band {
                (gradient, gradient, 30)
            } else {
                (30, gradient, gradient)
            }
        }
    }
}

impl FrameSource for MockSource {
    fn resolution(&self) -> Resolution {
        self.resolution
    }

    fn format(&self) -> PixelFormat {
        PixelFormat::Rgb8
    }

    fn next_frame(&mut self) -> Result<FrameBuffer, FrameiruError> {
        let sequence = self.sequence;
        self.sequence += 1;

        let (w, h) = (self.resolution.width, self.resolution.height);
        let len = PixelFormat::Rgb8.bytes_per_frame(self.resolution);
        let mut data = Vec::with_capacity(len);
        for y in 0..h {
            for x in 0..w {
                let (r, g, b) = Self::pattern(x, y, w, h, sequence);
                data.push(r);
                data.push(g);
                data.push(b);
            }
        }

        Ok(FrameBuffer {
            metadata: FrameMetadata {
                sequence,
                // Epoch-aligned so the pipeline's latency metric works.
                timestamp_us: timestamp_us_now(),
                resolution: self.resolution,
                format: PixelFormat::Rgb8,
            },
            data,
        })
    }
}

/// HSV (hue-angle only, full saturation/value) -> RGB for the test-pattern disc.
fn hsv_wheel(hue: f32) -> (u8, u8, u8) {
    let x = 1.0 - ((hue % 2.0) - 1.0).abs();
    let (r, g, b) = match (hue as u32) % 6 {
        0 => (1.0, x, 0.0),
        1 => (x, 1.0, 0.0),
        2 => (0.0, 1.0, x),
        3 => (0.0, x, 1.0),
        4 => (x, 0.0, 1.0),
        _ => (1.0, 0.0, x),
    };
    ((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn res() -> Resolution {
        Resolution {
            width: 64,
            height: 48,
        }
    }

    #[test]
    fn is_usable_as_frame_source() {
        let mut source = MockSource::new(res()).unwrap();
        let source: &mut dyn FrameSource = &mut source;
        assert_eq!(source.resolution(), res());
        assert_eq!(source.format(), PixelFormat::Rgb8);
        let frame = source.next_frame().unwrap();
        assert_eq!(frame.data.len(), PixelFormat::Rgb8.bytes_per_frame(res()));
    }

    #[test]
    fn sequence_increments_and_metadata_valid() {
        let mut source = MockSource::new(res()).unwrap();
        let f0 = source.next_frame().unwrap();
        let f1 = source.next_frame().unwrap();
        assert_eq!(f0.metadata.sequence, 0);
        assert_eq!(f1.metadata.sequence, 1);
        assert_eq!(source.sequence(), 2);
        assert_eq!(f0.metadata.format, PixelFormat::Rgb8);
    }

    #[test]
    fn consecutive_frames_differ() {
        let mut source = MockSource::new(res()).unwrap();
        let f0 = source.next_frame().unwrap();
        let f1 = source.next_frame().unwrap();
        let diff = f0
            .data
            .iter()
            .zip(f1.data.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert!(diff > 0, "animation must change between frames");
    }

    #[test]
    fn frame_has_variety() {
        let mut source = MockSource::new(res()).unwrap();
        let frame = source.next_frame().unwrap();
        let min = frame.data.iter().min().copied().unwrap_or(0);
        let max = frame.data.iter().max().copied().unwrap_or(0);
        assert!(max > min, "test pattern should not be a flat field");
    }

    #[test]
    fn rejects_invalid_resolution() {
        assert!(MockSource::new(Resolution {
            width: 0,
            height: 48,
        })
        .is_err());
    }
}
