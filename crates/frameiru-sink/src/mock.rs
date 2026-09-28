//! Validating in-memory sink for headless tests and benchmarks.
//!
//! Counts frames and bytes without touching hardware, optionally verifying
//! that frame data length matches its metadata.

use frameiru_core::error::FrameiruError;
use frameiru_core::format::FrameMetadata;
use frameiru_core::traits::FrameSink;
use frameiru_core::FrameBuffer;

/// Null sink that records what it receives.
///
/// When `validate` is set, `write_frame` rejects frames whose `data` length
/// does not match `metadata.bytes_len()` — a cheap correctness net for
/// pipeline stage integration tests.
pub struct MockSink {
    frames_written: usize,
    total_bytes: u64,
    last_metadata: Option<FrameMetadata>,
    validate: bool,
}

impl MockSink {
    /// Sink that records frames without validation.
    pub fn new() -> Self {
        Self::with_validation(false)
    }

    /// Sink that additionally checks data length against metadata.
    pub fn with_validation(validate: bool) -> Self {
        Self {
            frames_written: 0,
            total_bytes: 0,
            last_metadata: None,
            validate,
        }
    }

    pub fn frames_written(&self) -> usize {
        self.frames_written
    }

    pub fn total_bytes(&self) -> u64 {
        self.total_bytes
    }

    /// Metadata of the most recently written frame.
    pub fn last_metadata(&self) -> Option<&FrameMetadata> {
        self.last_metadata.as_ref()
    }
}

impl Default for MockSink {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameSink for MockSink {
    fn write_frame(&mut self, frame: &FrameBuffer) -> Result<(), FrameiruError> {
        if self.validate {
            let expected = frame.metadata.bytes_len();
            if frame.data.len() != expected {
                return Err(FrameiruError::InvalidArgument(format!(
                    "frame data length {} does not match metadata bytes_len() {expected}",
                    frame.data.len()
                )));
            }
        }
        self.frames_written += 1;
        self.total_bytes += frame.data.len() as u64;
        self.last_metadata = Some(frame.metadata);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use frameiru_capture::MockSource;
    use frameiru_core::format::{PixelFormat, Resolution};
    use frameiru_core::FrameSource;

    #[test]
    fn counts_frames_and_bytes() {
        let mut sink = MockSink::new();
        let mut source = MockSource::new(Resolution {
            width: 64,
            height: 48,
        })
        .unwrap();
        for _ in 0..5 {
            sink.write_frame(&source.next_frame().unwrap()).unwrap();
        }
        assert_eq!(sink.frames_written(), 5);
        assert_eq!(sink.total_bytes(), 5 * 64 * 48 * 3);
        assert_eq!(sink.last_metadata().unwrap().format, PixelFormat::Rgb8);
        assert_eq!(sink.last_metadata().unwrap().sequence, 4);
    }

    #[test]
    fn validation_rejects_mismatched_length() {
        let mut sink = MockSink::with_validation(true);
        let frame = FrameBuffer {
            metadata: frameiru_core::FrameMetadata {
                sequence: 0,
                timestamp_us: 0,
                resolution: Resolution {
                    width: 4,
                    height: 4,
                },
                format: PixelFormat::Rgb8,
            },
            data: vec![0u8; 4], // expects 4*4*3 = 48
        };
        assert!(matches!(
            sink.write_frame(&frame),
            Err(FrameiruError::InvalidArgument(_))
        ));
        assert_eq!(sink.frames_written(), 0);
    }

    #[test]
    fn no_validation_accepts_any_length() {
        let mut sink = MockSink::new();
        let frame = FrameBuffer {
            metadata: frameiru_core::FrameMetadata {
                sequence: 0,
                timestamp_us: 0,
                resolution: Resolution {
                    width: 4,
                    height: 4,
                },
                format: PixelFormat::Rgb8,
            },
            data: vec![0u8; 4],
        };
        sink.write_frame(&frame).unwrap();
        assert_eq!(sink.frames_written(), 1);
    }
}
