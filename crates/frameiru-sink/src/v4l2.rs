//! v4l2loopback virtual device writer (feature `v4l2`).
//!
//! [`LoopbackSink`] presents composited RGB8 frames as a virtual camera such
//! as `/dev/video10`: it converts each frame to YUYV (the format most video
//! apps expect from a loopback device) and streams it through the mmap
//! output interface of the `v4l` crate.
//!
//! # Send safety
//!
//! Like [`crate::v4l2`'s capture source](frameiru_capture::v4l2), the wrapped
//! [`v4l::io::mmap::Stream`] holds `&mut [u8]` borrows of kernel-mapped
//! buffers, so the struct cannot derive `Send` automatically. All access
//! happens through `FrameSink::write_frame(&mut self)`, so a moved sink is
//! only ever used from one thread; the `unsafe impl Send` encodes that
//! invariant.

use std::path::{Path, PathBuf};

use frameiru_core::error::FrameiruError;
use frameiru_core::format::{PixelFormat, Resolution};
use frameiru_core::traits::FrameSink;
use frameiru_core::FrameBuffer;
use v4l::buffer::Type;
use v4l::device::Device;
use v4l::format::FourCC;
use v4l::io::mmap::Stream;
use v4l::io::traits::OutputStream;
use v4l::video::traits::Output;

use crate::convert::rgb8_to_yuyv422;

const YUYV: FourCC = FourCC { repr: *b"YUYV" };

/// Writes RGB8 frames to a v4l2loopback device as YUYV 4:2:2.
pub struct LoopbackSink {
    stream: Stream<'static>,
    path: PathBuf,
    resolution: Resolution,
    /// Reused YUYV conversion buffer: one allocation for the sink's life.
    scratch: Vec<u8>,
}

impl LoopbackSink {
    /// Opens a loopback device (e.g. `/dev/video10`) and configures it for
    /// `resolution` YUYV output.
    pub fn open(path: impl AsRef<Path>, resolution: Resolution) -> Result<Self, FrameiruError> {
        let path = path.as_ref().to_path_buf();
        if !resolution.is_valid() {
            return Err(FrameiruError::InvalidArgument(format!(
                "loopback resolution must be non-zero, got {resolution:?}"
            )));
        }
        let device = Device::with_path(&path)?;

        let caps = device.query_caps()?;
        if !caps
            .capabilities
            .contains(v4l::capability::Flags::VIDEO_OUTPUT)
        {
            return Err(FrameiruError::Unsupported(format!(
                "{} is not a video output device",
                path.display()
            )));
        }

        let requested = v4l::Format::new(resolution.width, resolution.height, YUYV);
        let active = device.set_format(&requested).unwrap_or(requested);
        if active.fourcc != YUYV {
            return Err(FrameiruError::Unsupported(format!(
                "{} rejected YUYV output format",
                path.display()
            )));
        }

        let stream = Stream::new(&device, Type::VideoOutput)?;
        let scratch =
            vec![0u8; (resolution.width.div_ceil(2) as usize) * 4 * resolution.height as usize];

        Ok(Self {
            stream,
            path,
            resolution,
            scratch,
        })
    }

    /// Path of the opened loopback device.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Resolution the device was configured for.
    pub fn resolution(&self) -> Resolution {
        self.resolution
    }
}

// SAFETY: see the module-level "Send safety" section.
unsafe impl Send for LoopbackSink {}

impl FrameSink for LoopbackSink {
    fn write_frame(&mut self, frame: &FrameBuffer) -> Result<(), FrameiruError> {
        if frame.metadata.format != PixelFormat::Rgb8 {
            return Err(FrameiruError::FormatMismatch {
                expected: PixelFormat::Rgb8,
                actual: frame.metadata.format,
            });
        }
        if frame.metadata.resolution != self.resolution {
            return Err(FrameiruError::InvalidArgument(format!(
                "frame resolution {:?} does not match loopback resolution {:?}",
                frame.metadata.resolution, self.resolution
            )));
        }

        let (w, h) = (self.resolution.width, self.resolution.height);
        rgb8_to_yuyv422(&frame.data, w, h, &mut self.scratch)?;

        let (buf, meta) = self.stream.next()?;
        if buf.len() < self.scratch.len() {
            return Err(FrameiruError::InsufficientCapacity {
                need: self.scratch.len(),
                have: buf.len(),
            });
        }
        buf[..self.scratch.len()].copy_from_slice(&self.scratch);
        meta.bytesused = self.scratch.len() as u32;
        // Dropping the dequeued buffer hands it back to the kernel queue.
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_send<T: Send>() {}

    fn rgb_frame(resolution: Resolution, sequence: u64) -> FrameBuffer {
        FrameBuffer::new(frameiru_core::FrameMetadata {
            sequence,
            timestamp_us: 0,
            resolution,
            format: PixelFormat::Rgb8,
        })
    }

    #[test]
    fn sink_is_send() {
        assert_send::<LoopbackSink>();
    }

    #[test]
    fn open_missing_device_fails() {
        let res = Resolution {
            width: 640,
            height: 480,
        };
        assert!(LoopbackSink::open("/dev/video-this-does-not-exist", res).is_err());
    }

    #[test]
    fn open_rejects_zero_resolution() {
        let zero = Resolution {
            width: 0,
            height: 480,
        };
        assert!(LoopbackSink::open("/dev/video10", zero).is_err());
    }

    /// Hardware-gated smoke test: writes one frame when a loopback device is
    /// present (e.g. `modprobe v4l2loopback`).
    #[test]
    fn writes_a_frame_to_real_device() {
        let path =
            std::env::var("FRAMEIRU_LOOPBACK_DEVICE").unwrap_or_else(|_| "/dev/video10".into());
        if !Path::new(&path).exists() {
            eprintln!("skipping: {path} not present");
            return;
        }
        let res = Resolution {
            width: 64,
            height: 48,
        };
        let mut sink = match LoopbackSink::open(&path, res) {
            Ok(sink) => sink,
            Err(e) => {
                eprintln!("skipping: cannot open {path}: {e}");
                return;
            }
        };
        sink.write_frame(&rgb_frame(res, 0)).unwrap();
    }
}
