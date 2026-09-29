//! Frameiru segmentation engine.
//!
//! Turns camera frames into foreground masks: letterbox preprocessing
//! ([`preprocess`]), an optional ONNX model runner ([`model`], feature
//! `onnx`), unletterbox + bilinear postprocessing ([`postprocess`]), and an
//! EMA temporal smoother ([`smoother`]).

pub mod postprocess;
pub mod preprocess;
pub mod smoother;

#[cfg(feature = "onnx")]
pub mod download;
#[cfg(feature = "onnx")]
pub mod model;

pub use postprocess::postprocess_mask;
pub use preprocess::{preprocess_rgb8, Letterbox, Normalization};
pub use smoother::TemporalSmoother;

#[cfg(feature = "onnx")]
pub use model::{load_model, OnnxConfig, OnnxSegmenter, RvmSegmenter};
