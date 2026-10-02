//! Frameiru segmentation engine.
//!
//! Turns camera frames into foreground masks: letterbox preprocessing
//! ([`preprocess`]), an optional ONNX model runner ([`model`], feature
//! `onnx`), unletterbox + bilinear postprocessing ([`postprocess`]), and an
//! EMA temporal smoother ([`smoother`]).

pub mod fusion;
pub mod guided_filter;
pub mod polish;
pub mod postprocess;
pub mod preprocess;
pub mod roi;
pub mod smoother;

#[cfg(feature = "onnx")]
pub mod model;

pub use fusion::fuse_masks;
#[cfg(feature = "onnx")]
pub use fusion::FusionSegmenter;
pub use postprocess::{postprocess_mask, postprocess_mask_refined};
pub use preprocess::{preprocess_rgb8, Letterbox, Normalization};
pub use roi::{crop_frame, paste_mask_roi, RoiRect, RoiTracker};
pub use smoother::TemporalSmoother;

#[cfg(feature = "onnx")]
pub use model::{
    embedded_normalization, load_embedded, load_model, OnnxConfig, OnnxSegmenter, RvmSegmenter,
    EMBEDDED_MODEL_BYTES, EMBEDDED_MODEL_INPUT, MEDIAPIPE_MODEL_BYTES, RVM_MODEL_BYTES,
};
