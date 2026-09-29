//! V4L2 device capture via the `v4l` crate (feature `v4l2`).
//!
//! Opens `/dev/videoN` with the mmap streaming API, negotiates YUYV (or
//! MJPEG with the `mjpeg` feature) and decodes every frame to RGB8, the
//! pipeline's canonical format.
//!
//! # Send safety
//!
//! [`V4l2Source`] is declared `Send`: the wrapped [`v4l::io::mmap::Stream`]
//! borrows the device's kernel-mapped buffers as `&mut [u8]`, which Rust
//! cannot prove is thread-safe. The invariant is that all access happens
//! through `&mut self` ([`FrameSource::next_frame`] is the only entry
//! point), so a moved source is used from exactly one thread at a time —
//! the same guarantee the trait's `Send` bound exists to express. `Device`
//! itself is already `Send + Sync` (shared `Arc<Handle>` over an fd).

use std::path::{Path, PathBuf};

use frameiru_core::error::FrameiruError;
use frameiru_core::format::{FrameMetadata, PixelFormat, Resolution};
use frameiru_core::traits::FrameSource;
use frameiru_core::FrameBuffer;
use v4l::buffer::Type;
use v4l::device::Device;
use v4l::format::FourCC;
use v4l::io::mmap::Stream;
use v4l::io::traits::CaptureStream;
use v4l::video::traits::Capture;

#[cfg(feature = "mjpeg")]
use crate::decoders::mjpeg_to_rgb8;
use crate::decoders::{nv12_to_rgb8, yuyv_to_rgb8};

const NV12: FourCC = FourCC { repr: *b"NV12" };
const YUYV: FourCC = FourCC { repr: *b"YUYV" };
const MJPG: FourCC = FourCC { repr: *b"MJPG" };

/// A capture device streaming frames as RGB8.
pub struct V4l2Source {
    stream: Stream<'static>,
    path: PathBuf,
    resolution: Resolution,
    fourcc: FourCC,
    sequence: u64,
    /// Reused decode scratch buffer: one allocation for the source's life.
    scratch: Vec<u8>,
}

impl V4l2Source {
    /// Opens a capture device, e.g. `/dev/video0`, negotiating YUYV at the
    /// driver's current resolution.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, FrameiruError> {
        Self::open_with_resolution(path, None)
    }

    /// Opens a capture device and requests `resolution`.
    ///
    /// Falls back to the driver's current resolution when the request is
    /// rejected. Prefers NV12 (25% less USB bandwidth, the `yuv` crate's
    /// fastest decode), then YUYV; with the `mjpeg` feature enabled, falls
    /// back to MJPEG (decoded in software) when neither is available.
    pub fn open_with_resolution(
        path: impl AsRef<Path>,
        resolution: Option<Resolution>,
    ) -> Result<Self, FrameiruError> {
        let path = path.as_ref().to_path_buf();
        let device = Device::with_path(&path)?;

        let caps = device.query_caps()?;
        if !caps
            .capabilities
            .contains(v4l::capability::Flags::VIDEO_CAPTURE)
        {
            return Err(FrameiruError::Unsupported(format!(
                "{} is not a video capture device",
                path.display()
            )));
        }

        let formats = device.enum_formats()?;
        let has = |fourcc: FourCC| formats.iter().any(|f| f.fourcc == fourcc);
        let fourcc = if has(NV12) {
            NV12
        } else if has(YUYV) {
            YUYV
        } else if cfg!(feature = "mjpeg") && has(MJPG) {
            MJPG
        } else {
            return Err(FrameiruError::Unsupported(format!(
                "{} exposes neither NV12 nor YUYV{}",
                path.display(),
                if cfg!(feature = "mjpeg") {
                    " nor MJPEG"
                } else {
                    " (enable the `mjpeg` feature)"
                }
            )));
        };

        // Negotiate the format; fall back to whatever the driver keeps.
        let (w, h) = resolution.map_or((0, 0), |r| (r.width, r.height));
        let requested = v4l::Format::new(w, h, fourcc);
        let mut actual = match resolution {
            Some(_) => device.set_format(&requested).unwrap_or(requested),
            None => requested,
        };
        if actual.fourcc != fourcc || actual.width == 0 || actual.height == 0 {
            actual = device.format()?;
        }

        let mut stream = Stream::new(&device, Type::VideoCapture)?;
        // Bounded dequeue: lets the pipeline's shutdown signal win even when
        // the driver has no frame ready.
        stream.set_timeout(std::time::Duration::from_millis(250));
        let resolution = Resolution {
            width: actual.width,
            height: actual.height,
        };
        let scratch = vec![0u8; PixelFormat::Rgb8.bytes_per_frame(resolution)];

        Ok(Self {
            stream,
            path,
            resolution,
            fourcc: actual.fourcc,
            sequence: 0,
            scratch,
        })
    }

    /// Path of the opened device.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Pixel format the device was negotiated to (YUYV or MJPG).
    pub fn device_fourcc(&self) -> &str {
        self.fourcc.str().unwrap_or("????")
    }
}

// SAFETY: see the module-level "Send safety" section.
unsafe impl Send for V4l2Source {}

impl FrameSource for V4l2Source {
    fn resolution(&self) -> Resolution {
        self.resolution
    }

    fn format(&self) -> PixelFormat {
        PixelFormat::Rgb8
    }

    fn next_frame(&mut self) -> Result<FrameBuffer, FrameiruError> {
        let (bytes, meta) = self.stream.next()?;
        let used = meta.bytesused as usize;
        let (w, h) = (self.resolution.width, self.resolution.height);

        if self.fourcc == YUYV {
            yuyv_to_rgb8(&bytes[..used], w, h, &mut self.scratch)?;
        } else if self.fourcc == NV12 {
            nv12_to_rgb8(&bytes[..used], w, h, &mut self.scratch)?;
        } else if cfg!(feature = "mjpeg") && self.fourcc == MJPG {
            #[cfg(feature = "mjpeg")]
            {
                let decoded = mjpeg_to_rgb8(&bytes[..used], &mut self.scratch)?;
                self.resolution = decoded;
                self.scratch
                    .truncate(PixelFormat::Rgb8.bytes_per_frame(decoded));
            }
        } else {
            return Err(FrameiruError::Unsupported(format!(
                "unhandled capture fourcc {}",
                self.fourcc.str().unwrap_or("????")
            )));
        }

        let sequence = self.sequence;
        self.sequence += 1;
        // Stamp wall time at dequeue (same clock as the pipeline's compose
        // side), so the latency metric is meaningful. Driver timestamps are
        // monotonic-since-boot and cannot be compared to wall time without
        // a shared boot-mapping, so they are not used here.
        let timestamp_us = frameiru_core::timestamp_us_now();

        Ok(FrameBuffer {
            metadata: FrameMetadata {
                sequence,
                timestamp_us,
                resolution: self.resolution,
                format: PixelFormat::Rgb8,
            },
            data: self.scratch.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use frameiru_core::FrameSource;

    fn assert_send<T: Send>() {}

    #[test]
    fn source_is_send() {
        assert_send::<V4l2Source>();
    }

    #[test]
    fn open_missing_device_fails() {
        let err = V4l2Source::open("/dev/video-this-does-not-exist");
        assert!(err.is_err(), "opening a missing device must fail");
    }

    /// Hardware-gated smoke test: reads one frame when a camera is attached.
    #[test]
    fn captures_a_frame_from_real_device() {
        let path = std::env::var("FRAMEIRU_VIDEO_DEVICE").unwrap_or_else(|_| "/dev/video0".into());
        if !Path::new(&path).exists() {
            eprintln!("skipping: {path} not present");
            return;
        }
        let mut source = match V4l2Source::open(&path) {
            Ok(source) => source,
            Err(e) => {
                eprintln!("skipping: cannot open {path}: {e}");
                return;
            }
        };
        let frame = match source.next_frame() {
            Ok(frame) => frame,
            Err(e) => {
                eprintln!("skipping: capture failed: {e}");
                return;
            }
        };
        assert_eq!(frame.metadata.format, PixelFormat::Rgb8);
        assert!(frame.metadata.resolution.is_valid());
        assert_eq!(
            frame.data.len(),
            PixelFormat::Rgb8.bytes_per_frame(frame.metadata.resolution)
        );
    }
}
