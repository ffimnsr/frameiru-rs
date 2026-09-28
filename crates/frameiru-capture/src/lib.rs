//! Frameiru video capture crate.
//!
//! Phase 2 provides a headless synthetic camera ([`MockSource`]) and pixel
//! format decoding helpers ([`yuyv_to_rgb8`], optional MJPEG decoder). The
//! real V4L2 device streamer lives in [`v4l2`] and is feature-gated behind
//! `v4l2` so workspace builds and tests work without a camera attached.

pub mod decoders;
pub mod mock;

#[cfg(feature = "v4l2")]
pub mod v4l2;

#[cfg(feature = "v4l2")]
pub use v4l;

pub use decoders::yuyv_to_rgb8;
pub use mock::MockSource;

#[cfg(feature = "v4l2")]
pub use v4l2::V4l2Source;
