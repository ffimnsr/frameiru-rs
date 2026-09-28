//! ONNX model runner via the `ort` crate (feature `onnx`).
//!
//! [`OnnxSegmenter`] loads a portrait-matting ONNX graph, feeds letterboxed
//! normalized RGB8 frames in as a `1x3xHxW` float tensor, and unletterboxes
//! the mask output back to the frame resolution.

use std::path::Path;

use frameiru_core::buffer::Mask;
use frameiru_core::error::FrameiruError;
use frameiru_core::format::{PixelFormat, Resolution};
use frameiru_core::traits::Segmenter;
use frameiru_core::FrameBuffer;
use ort::session::Session;
use ort::value::{Tensor, TensorValueType};

use crate::postprocess::postprocess_mask;
use crate::preprocess::{preprocess_rgb8, Normalization};

/// Options for loading a segmentation model.
#[derive(Debug, Clone, PartialEq)]
pub struct OnnxConfig {
    /// Fixed input canvas the model expects (e.g. 320x320).
    pub input_size: Resolution,
    pub normalization: Normalization,
    /// Model input tensor name.
    pub input_name: String,
    /// Model output tensor name.
    pub output_name: String,
}

impl OnnxConfig {
    pub fn new(input_size: Resolution) -> Result<Self, FrameiruError> {
        if !input_size.is_valid() {
            return Err(FrameiruError::InvalidArgument(
                "model input size must be non-zero".into(),
            ));
        }
        Ok(Self {
            input_size,
            normalization: Normalization::default(),
            input_name: "input".into(),
            output_name: "output".into(),
        })
    }
}

/// A loaded ONNX segmentation model implementing [`Segmenter`].
pub struct OnnxSegmenter {
    session: Session,
    config: OnnxConfig,
}

impl OnnxSegmenter {
    /// Loads `path` (a `.onnx` file) with `config`.
    pub fn load(path: impl AsRef<Path>, config: OnnxConfig) -> Result<Self, FrameiruError> {
        let mut builder = Session::builder()
            .map_err(|e| FrameiruError::Segmentation(format!("ort init failed: {e}")))?;
        let session = builder.commit_from_file(path.as_ref()).map_err(|e| {
            FrameiruError::Segmentation(format!(
                "failed to load model {}: {e}",
                path.as_ref().display()
            ))
        })?;
        Ok(Self { session, config })
    }

    pub fn config(&self) -> &OnnxConfig {
        &self.config
    }
}

impl Segmenter for OnnxSegmenter {
    fn input_resolution(&self) -> Resolution {
        self.config.input_size
    }

    fn segment(&mut self, frame: &FrameBuffer) -> Result<Mask, FrameiruError> {
        if frame.metadata.format != PixelFormat::Rgb8 {
            return Err(FrameiruError::FormatMismatch {
                expected: PixelFormat::Rgb8,
                actual: frame.metadata.format,
            });
        }
        let (iw, ih) = (
            self.config.input_size.width as usize,
            self.config.input_size.height as usize,
        );
        let area = iw * ih;

        let mut tensor = vec![0f32; 3 * area];
        let letterbox = preprocess_rgb8(
            &frame.data,
            frame.metadata.resolution,
            self.config.input_size,
            self.config.normalization,
            &mut tensor,
        )?;

        let input = Tensor::<f32>::from_array(([1usize, 3, ih, iw], tensor))
            .map_err(|e| FrameiruError::Segmentation(format!("tensor build failed: {e}")))?;
        let input_name = self.config.input_name.clone();
        let outputs = self
            .session
            .run(ort::inputs! { input_name => input })
            .map_err(|e| FrameiruError::Segmentation(format!("inference failed: {e}")))?;
        let output = outputs
            .get(self.config.output_name.as_str())
            .ok_or_else(|| {
                FrameiruError::Segmentation(format!(
                    "model has no output named '{}'",
                    self.config.output_name
                ))
            })?;
        let tensor_ref = output
            .downcast_ref::<TensorValueType<f32>>()
            .map_err(|e| FrameiruError::Segmentation(format!("output is not f32: {e}")))?;
        let (_, data) = tensor_ref.extract_tensor();

        // Outputs are [1,1,H,W] (or [1,H,W]); the mask is the trailing H*W.
        if data.len() < area {
            return Err(FrameiruError::Segmentation(format!(
                "model output has {} values, expected at least {area}",
                data.len()
            )));
        }
        let model_mask = Mask {
            resolution: self.config.input_size,
            data: data[data.len() - area..].to_vec(),
        };

        let mut out = Mask::filled(frame.metadata.resolution, 0.0);
        postprocess_mask(&model_mask, frame.metadata.resolution, &letterbox, &mut out)?;
        Ok(out)
    }
}
