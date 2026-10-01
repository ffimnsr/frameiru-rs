//! Frameiru pipeline engine.
//!
//! Wires capture, segmentation, and composition stages into a multi-threaded
//! pipeline ([`Engine`]) with a cloneable control handle ([`PipelineHandle`]):
//! dynamic background updates, a preview broadcast, pacing, metrics, and
//! clean shutdown.

mod config;
mod engine;
mod metrics;
mod motion;
mod runner;

pub use config::PipelineConfig;
pub use engine::{Engine, PipelineHandle};
pub use metrics::MetricsSnapshot;
