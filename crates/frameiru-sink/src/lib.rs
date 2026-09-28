//! Frameiru output sink crate.
//!
//! Phase 2 implements the outputs of the pipeline:
//! - [`BroadcastSink`]: in-memory preview channel for UI/IPC monitors.
//! - [`MockSink`]: validating sink for headless tests and benchmarks.
//! - [`rgb8_to_yuyv422`]: scalar RGB24 -> YUYV422 conversion (BT.601).
//! - `v4l2` (feature): v4l2loopback virtual device writer.

pub mod broadcast;
pub mod convert;
pub mod mock;

#[cfg(feature = "v4l2")]
pub mod v4l2;

pub use broadcast::BroadcastSink;
pub use convert::rgb8_to_yuyv422;
pub use mock::MockSink;
