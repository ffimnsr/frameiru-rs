//! Frameiru core primitives.
//!
//! Shared types and traits that all other crates build on: pixel formats,
//! resolutions, zero-allocation frame buffer pooling, segmentation masks,
//! background modes, the central error type, and the engine boundary traits
//! ([`FrameSource`], [`Segmenter`], [`Compositor`], [`FrameSink`]).

#![forbid(unsafe_code)]

pub mod buffer;
pub mod error;
pub mod format;
pub mod mode;
pub mod time;
pub mod traits;

#[cfg(feature = "slint")]
pub mod slint_compat;

pub use buffer::{BufferPool, FrameBuffer, Mask, PooledBuffer};
pub use error::FrameiruError;
pub use format::{FrameMetadata, PixelFormat, Resolution};
pub use mode::BackgroundMode;
pub use time::timestamp_us_now;
pub use traits::{Compositor, FrameSink, FrameSource, Segmenter};
