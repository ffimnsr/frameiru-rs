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
    /// Resolved tensor names: configured value when present in the graph,
    /// otherwise the graph's first input/output (models such as u2net name
    /// their output "d1").
    input_name: String,
    output_name: String,
    /// Reused CHW preprocessing buffer; the input tensor views into it.
    tensor_buf: Vec<f32>,
}

impl OnnxSegmenter {
    /// Loads `path` (a `.onnx` file) with `config`.
    pub fn load(path: impl AsRef<Path>, config: OnnxConfig) -> Result<Self, FrameiruError> {
        let mut builder = Session::builder()
            .map_err(|e| FrameiruError::Segmentation(format!("ort init failed: {e}")))?;
        builder = builder
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
            .map_err(|e| FrameiruError::Segmentation(format!("optimization setup failed: {e}")))?;
        let session = builder.commit_from_file(path.as_ref()).map_err(|e| {
            FrameiruError::Segmentation(format!(
                "failed to load model {}: {e}",
                path.as_ref().display()
            ))
        })?;
        let resolve =
            |configured: &str, graph: &[ort::value::Outlet]| -> Result<String, FrameiruError> {
                graph
                    .iter()
                    .find(|o| o.name() == configured)
                    .or_else(|| graph.first())
                    .map(|o| o.name().to_string())
                    .ok_or_else(|| {
                        FrameiruError::Segmentation(format!(
                            "model has no tensors (input/output) named '{configured}', "
                        ))
                    })
            };
        let input_name = resolve(&config.input_name, session.inputs())?;
        let output_name = resolve(&config.output_name, session.outputs())?;
        if input_name != config.input_name || output_name != config.output_name {
            tracing::info!("model tensors resolved: input '{input_name}', output '{output_name}'");
        }
        Ok(Self {
            session,
            config,
            input_name,
            output_name,
            // Reused preprocessing buffer: the input tensor views into this
            // (zero-copy), so steady-state inference performs no per-frame
            // tensor allocation or copy.
            tensor_buf: Vec::new(),
        })
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

        self.tensor_buf.resize(3 * area, 0.0);
        let letterbox = preprocess_rgb8(
            &frame.data,
            frame.metadata.resolution,
            self.config.input_size,
            self.config.normalization,
            &mut self.tensor_buf,
        )?;

        // Zero-copy input: the tensor borrows the preprocessing buffer.
        let input = ort::value::TensorRef::<f32>::from_array_view((
            [1usize, 3, ih, iw],
            self.tensor_buf.as_slice(),
        ))
        .map_err(|e| FrameiruError::Segmentation(format!("tensor view failed: {e}")))?;
        let input_name = self.input_name.clone();
        let outputs = self
            .session
            .run(ort::inputs! { input_name => input })
            .map_err(|e| FrameiruError::Segmentation(format!("inference failed: {e}")))?;
        let output = outputs.get(self.output_name.as_str()).ok_or_else(|| {
            FrameiruError::Segmentation(format!("model has no output named '{}'", self.output_name))
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

/// Hidden-state channel counts for the RVM MobileNetV3 export (official
/// `rvm_mobilenetv3_fp32.onnx`: `r1i..r4i` of `(1, dim, 1, 1)`).
const RVM_STATE_DIMS: [i64; 4] = [16, 20, 40, 64];
const RVM_STATE_INPUTS: [&str; 4] = ["r1i", "r2i", "r3i", "r4i"];
const RVM_STATE_OUTPUTS: [&str; 4] = ["r1o", "r2o", "r3o", "r4o"];

/// Robust Video Matting (MobileNetV3): a recurrent segmenter.
///
/// Feeds the previous frame's hidden states (`r1i..r4i`) back in on every
/// call and stores the new states (`r1o..r4o`), which gives temporally
/// coherent mattes. [`Segmenter::reset_state`] zeroes the states; the
/// pipeline calls it when capture frames are dropped so a discontinuity
/// cannot ghost.
pub struct RvmSegmenter {
    session: Session,
    config: OnnxConfig,
    input_name: String,
    state_inputs: [String; 4],
    state_outputs: [String; 4],
    output_name: String,
    /// Scalar control input (`downsample_ratio`): 1.0 at 256x256.
    ratio_input: Option<String>,
    states: [Vec<f32>; 4],
}

impl RvmSegmenter {
    /// Loads an RVM ONNX graph (input `src`, states `r*i`/`r*o`, mask `pha`).
    /// Falls back to positional tensor lookup when names differ.
    pub fn load(path: impl AsRef<Path>, config: OnnxConfig) -> Result<Self, FrameiruError> {
        let mut builder = Session::builder()
            .map_err(|e| FrameiruError::Segmentation(format!("ort init failed: {e}")))?;
        builder = builder
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
            .map_err(|e| FrameiruError::Segmentation(format!("optimization setup failed: {e}")))?;
        let session = builder.commit_from_file(path.as_ref()).map_err(|e| {
            FrameiruError::Segmentation(format!(
                "failed to load model {}: {e}",
                path.as_ref().display()
            ))
        })?;

        let inputs = session.inputs();
        let outputs = session.outputs();
        let name_at = |slice: &[ort::value::Outlet], i: usize| -> Option<String> {
            slice.get(i).map(|o| o.name().to_string())
        };
        let named = |slice: &[ort::value::Outlet], want: &str, fallback: usize| -> Option<String> {
            slice
                .iter()
                .find(|o| o.name() == want)
                .or_else(|| slice.get(fallback))
                .map(|o| o.name().to_string())
        };

        let input_name = name_at(inputs, 0)
            .ok_or_else(|| FrameiruError::Segmentation("RVM graph has no inputs".into()))?;
        let mut state_inputs = [String::new(), String::new(), String::new(), String::new()];
        let mut state_outputs = [String::new(), String::new(), String::new(), String::new()];
        for i in 0..4 {
            state_inputs[i] = named(inputs, RVM_STATE_INPUTS[i], 1 + i).ok_or_else(|| {
                FrameiruError::Segmentation(format!(
                    "RVM missing state input {}",
                    RVM_STATE_INPUTS[i]
                ))
            })?;
            state_outputs[i] = named(outputs, RVM_STATE_OUTPUTS[i], outputs.len() - 4 + i)
                .ok_or_else(|| {
                    FrameiruError::Segmentation(format!(
                        "RVM missing state output {}",
                        RVM_STATE_OUTPUTS[i]
                    ))
                })?;
        }
        let output_name = named(outputs, "pha", 1)
            .ok_or_else(|| FrameiruError::Segmentation("RVM graph has no mask output".into()))?;
        // The official export also takes a scalar `downsample_ratio`;
        // 1.0 at the small 256x256 input we feed.
        let ratio_input = inputs
            .iter()
            .find(|o| o.name() == "downsample_ratio")
            .or_else(|| inputs.get(5))
            .map(|o| o.name().to_string());

        let states = RVM_STATE_DIMS.map(|d| vec![0.0; d as usize]);
        tracing::info!(
            "RVM resolved: src '{input_name}', pha '{output_name}', states {state_inputs:?} -> {state_outputs:?}"
        );
        Ok(Self {
            session,
            config,
            input_name,
            state_inputs,
            state_outputs,
            output_name,
            ratio_input,
            states,
        })
    }

    /// Debug/stateful-testing hook: checksum of the current hidden states.
    pub fn state_checksum(&self) -> u64 {
        self.states
            .iter()
            .flat_map(|s| s.iter())
            .fold(0u64, |acc, v| {
                acc.wrapping_add((v.to_bits() as u64).rotate_left(13))
            })
    }

    pub fn config(&self) -> &OnnxConfig {
        &self.config
    }
}

impl Segmenter for RvmSegmenter {
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

        let src = Tensor::<f32>::from_array(([1usize, 3, ih, iw], tensor))
            .map_err(|e| FrameiruError::Segmentation(format!("tensor build failed: {e}")))?;
        let state_tensors = self
            .states
            .iter()
            .zip(RVM_STATE_DIMS)
            .map(|(state, dim)| {
                Tensor::<f32>::from_array(([1usize, dim as usize, 1, 1], state.clone()))
                    .map_err(|e| FrameiruError::Segmentation(format!("state tensor failed: {e}")))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let input_name = self.input_name.clone();
        let s0 = self.state_inputs[0].clone();
        let s1 = self.state_inputs[1].clone();
        let s2 = self.state_inputs[2].clone();
        let s3 = self.state_inputs[3].clone();
        let outputs = match &self.ratio_input {
            Some(ratio_name) => {
                let ratio = Tensor::<f32>::from_array(([1usize], vec![1.0f32])).map_err(|e| {
                    FrameiruError::Segmentation(format!("ratio tensor failed: {e}"))
                })?;
                let ratio_name = ratio_name.clone();
                self.session
                    .run(ort::inputs! {
                        input_name => src,
                        s0 => state_tensors[0].clone(),
                        s1 => state_tensors[1].clone(),
                        s2 => state_tensors[2].clone(),
                        s3 => state_tensors[3].clone(),
                        ratio_name => ratio,
                    })
                    .map_err(|e| FrameiruError::Segmentation(format!("inference failed: {e}")))?
            }
            None => self
                .session
                .run(ort::inputs! {
                    input_name => src,
                    s0 => state_tensors[0].clone(),
                    s1 => state_tensors[1].clone(),
                    s2 => state_tensors[2].clone(),
                    s3 => state_tensors[3].clone(),
                })
                .map_err(|e| FrameiruError::Segmentation(format!("inference failed: {e}")))?,
        };

        let mask_val = outputs.get(self.output_name.as_str()).ok_or_else(|| {
            FrameiruError::Segmentation(format!("model has no output named '{}'", self.output_name))
        })?;
        let tensor_ref = mask_val
            .downcast_ref::<TensorValueType<f32>>()
            .map_err(|e| FrameiruError::Segmentation(format!("mask output is not f32: {e}")))?;
        let (_, data) = tensor_ref.extract_tensor();
        if data.len() < area {
            return Err(FrameiruError::Segmentation(format!(
                "RVM mask output has {} values, expected at least {area}",
                data.len()
            )));
        }
        let model_mask = Mask {
            resolution: self.config.input_size,
            data: data[data.len() - area..].to_vec(),
        };

        // Roll the recurrent states forward.
        for (slot, name) in self.state_outputs.iter().enumerate() {
            let val = outputs.get(name.as_str()).ok_or_else(|| {
                FrameiruError::Segmentation(format!("RVM missing state output '{name}'"))
            })?;
            let state_ref = val
                .downcast_ref::<TensorValueType<f32>>()
                .map_err(|e| FrameiruError::Segmentation(format!("state is not f32: {e}")))?;
            let (_, state_data) = state_ref.extract_tensor();
            let want = self.states[slot].len();
            if state_data.len() < want {
                return Err(FrameiruError::Segmentation(format!(
                    "RVM state '{name}' has {} values, expected {want}",
                    state_data.len()
                )));
            }
            self.states[slot].copy_from_slice(&state_data[..want]);
        }

        let mut out = Mask::filled(frame.metadata.resolution, 0.0);
        postprocess_mask(&model_mask, frame.metadata.resolution, &letterbox, &mut out)?;
        Ok(out)
    }

    fn reset_state(&mut self) {
        for state in &mut self.states {
            state.fill(0.0);
        }
        tracing::debug!("RVM recurrent state reset");
    }
}

/// Loads `path`, auto-detecting RVM (>= 5 inputs with recurrent states)
/// vs. a plain single-input model.
pub fn load_model(
    path: impl AsRef<Path>,
    config: OnnxConfig,
) -> Result<Box<dyn Segmenter>, FrameiruError> {
    let path = path.as_ref();
    // Probe the graph once; the chosen segmenter reopens it (startup only).
    let mut builder = Session::builder()
        .map_err(|e| FrameiruError::Segmentation(format!("ort init failed: {e}")))?;
    let probe = builder.commit_from_file(path).map_err(|e| {
        FrameiruError::Segmentation(format!("failed to load model {}: {e}", path.display()))
    })?;
    if probe.inputs().len() >= 5 {
        Ok(Box::new(RvmSegmenter::load(path, config)?))
    } else {
        Ok(Box::new(OnnxSegmenter::load(path, config)?))
    }
}
