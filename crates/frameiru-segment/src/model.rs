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
use ort::value::TensorValueType;

use crate::guided_filter::{DEFAULT_GUIDED_EPS, DEFAULT_GUIDED_RADIUS};
use crate::postprocess::{postprocess_mask, postprocess_mask_refined};
use crate::preprocess::{preprocess_rgb8, Normalization};

/// Bytes of the embedded MediaPipe selfie model (~462 KB).
pub const MEDIAPIPE_MODEL_BYTES: &[u8] = include_bytes!("../assets/selfie_segmentation.onnx");

/// Bytes of the embedded RVM MobileNetV3 model (~15 MB).
pub const RVM_MODEL_BYTES: &[u8] = include_bytes!("../assets/rvm_mobilenetv3_fp32.onnx");

/// Default embedded primary bytes (MediaPipe anchor).
pub const EMBEDDED_MODEL_BYTES: &[u8] = MEDIAPIPE_MODEL_BYTES;

/// Input canvas of the embedded model.
pub const EMBEDDED_MODEL_INPUT: Resolution = Resolution {
    width: 256,
    height: 256,
};

/// Normalization of the embedded model: plain `v / 255` ([0, 1]).
pub fn embedded_normalization() -> Normalization {
    Normalization::unit()
}

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
    /// Refine the upsampled mask with a guided filter guided by the full-res
    /// frame (U9.4): soft matte edges snap to sharp RGB boundaries. Off for
    /// raw throughput benchmarks; fast-GF path ~2-3 ms at 640x480 release.
    pub refine_mask: bool,
    /// Guided-filter neighborhood radius (half-size, pixels).
    pub guided_radius: u32,
    /// Guided-filter regularization; luma variance below this is flat.
    pub guided_eps: f32,
    /// ORT intra-op thread cap; `None` pins the session to the physical core
    /// count (spawning all logical cores rarely helps latency and heats the
    /// CPU, U9.6).
    pub intra_threads: Option<usize>,
    /// Foreground mask dilation (3x3 max filter, iterations) before
    /// compositing: grows the subject by a hair so missed background
    /// regions at the matte edge cannot leak through as sharp pixels.
    /// 0 disables. Cheap (~0.3 ms at 640x480 per pass).
    pub mask_dilate: u32,
    /// Soft mask threshold center: `smoothstep(center-0.15, center+0.15)`
    /// collapses ghosting mid-alphas toward 0/1 (less "partially-sharp"
    /// fringe). `0.0` disables.
    pub mask_contrast: f32,
    /// Dynamic crop & track (ROI zoom) around the subject. Off by default
    /// because cropping to a bounding box causes delay and boundary clipping
    /// during rapid movement.
    pub roi_zoom: bool,
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
            normalization: Normalization::unit(),
            input_name: "input".into(),
            output_name: "output".into(),
            refine_mask: true,
            guided_radius: DEFAULT_GUIDED_RADIUS,
            guided_eps: DEFAULT_GUIDED_EPS,
            intra_threads: None,
            mask_dilate: 0,
            mask_contrast: 0.0,
            roi_zoom: false,
        })
    }
}

/// Builds an ORT session builder with graph optimization plus the intra-op
/// thread cap: an explicit `config.intra_threads` (U9.6), or pinned to the
/// physical core count — spawning one ORT thread per logical core rarely
/// helps latency and wastes power on SMT (e.g. 12 threads on a 5600X vs 6).
fn session_builder(
    config: &OnnxConfig,
) -> Result<ort::session::builder::SessionBuilder, FrameiruError> {
    let mut builder = Session::builder()
        .map_err(|e| FrameiruError::Segmentation(format!("ort init failed: {e}")))?;
    builder = builder
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
        .map_err(|e| FrameiruError::Segmentation(format!("optimization setup failed: {e}")))?;
    let threads = match config.intra_threads {
        Some(0) => {
            return Err(FrameiruError::InvalidArgument(
                "intra_threads must be >= 1; omit it to pin to physical cores".into(),
            ));
        }
        Some(n) => n,
        None => physical_core_count(),
    };
    builder = builder
        .with_intra_threads(threads)
        .map_err(|e| FrameiruError::Segmentation(format!("intra-thread setup failed: {e}")))?;
    Ok(builder)
}

/// Physical core count: unique `(package, core)` pairs from sysfs on Linux,
/// falling back to logical parallelism elsewhere.
fn physical_core_count() -> usize {
    #[cfg(target_os = "linux")]
    if let Some(n) = sysfs_physical_cores() {
        return n;
    }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// Unique `(package, core)` pairs from `/sys/devices/system/cpu`, or `None`
/// when sysfs is unavailable (containers, non-Linux).
#[cfg(target_os = "linux")]
fn sysfs_physical_cores() -> Option<usize> {
    let base = std::path::Path::new("/sys/devices/system/cpu");
    let online = std::fs::read_to_string(base.join("online")).ok()?;
    let mut ids = std::collections::HashSet::new();
    for id in parse_cpu_list(&online) {
        let pkg =
            std::fs::read_to_string(base.join(format!("cpu{id}/topology/physical_package_id")))
                .ok()?
                .trim()
                .parse::<u32>()
                .ok()?;
        let core = std::fs::read_to_string(base.join(format!("cpu{id}/topology/core_id")))
            .ok()?
            .trim()
            .parse::<u32>()
            .ok()?;
        ids.insert((pkg, core));
    }
    if ids.is_empty() {
        None
    } else {
        Some(ids.len())
    }
}

/// Parses a Linux CPU-list (`"0-3,7,10-12"`) into ids; unrecognized tokens
/// (e.g. `"0-7:2"` step syntax) are skipped.
fn parse_cpu_list(list: &str) -> Vec<u32> {
    let mut out = Vec::new();
    for token in list.split(',') {
        let token = token.trim();
        if let Some((lo, hi)) = token.split_once('-') {
            let Ok(lo) = lo.trim().parse::<u32>() else {
                continue;
            };
            let Ok(hi) = hi.trim().parse::<u32>() else {
                continue;
            };
            out.extend(lo..=hi);
        } else if let Ok(id) = token.parse::<u32>() {
            out.push(id);
        }
    }
    out
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
        let bytes = std::fs::read(path.as_ref()).map_err(|e| {
            FrameiruError::Segmentation(format!(
                "cannot read model {}: {e}",
                path.as_ref().display()
            ))
        })?;
        Self::load_bytes(&bytes, config)
    }

    /// Loads a model from an in-memory buffer (e.g. the embedded default).
    pub fn load_bytes(bytes: &[u8], config: OnnxConfig) -> Result<Self, FrameiruError> {
        let mut builder = session_builder(&config)?;
        let session = builder.commit_from_memory(bytes).map_err(|e| {
            FrameiruError::Segmentation(format!("failed to load model from memory: {e}"))
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

        let mut out = Mask::default();
        if self.config.refine_mask {
            postprocess_mask_refined(
                &model_mask,
                frame.metadata.resolution,
                &letterbox,
                &frame.data,
                self.config.guided_radius,
                self.config.guided_eps,
                &mut out,
            )?;
        } else {
            postprocess_mask(&model_mask, frame.metadata.resolution, &letterbox, &mut out)?;
        }
        crate::polish::polish_mask(&mut out, self.config.mask_dilate, self.config.mask_contrast)?;
        Ok(out)
    }
}

/// Hidden-state channel counts for the RVM MobileNetV3 export (official
/// `rvm_mobilenetv3_fp32.onnx`: `r1i..r4i` of `(1, dim, 1, 1)`).
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
    states: [Option<(Vec<usize>, Vec<f32>)>; 4],
    zero_states: [(Vec<usize>, Vec<f32>); 4],
    tensor_buf: Vec<f32>,
}

impl RvmSegmenter {
    /// Loads an RVM ONNX graph (input `src`, states `r*i`/`r*o`, mask `pha`).
    /// Falls back to positional tensor lookup when names differ.
    pub fn load(path: impl AsRef<Path>, config: OnnxConfig) -> Result<Self, FrameiruError> {
        let bytes = std::fs::read(path.as_ref()).map_err(|e| {
            FrameiruError::Segmentation(format!(
                "cannot read model {}: {e}",
                path.as_ref().display()
            ))
        })?;
        Self::load_bytes(&bytes, config)
    }

    /// Loads an RVM ONNX graph from an in-memory buffer.
    pub fn load_bytes(bytes: &[u8], config: OnnxConfig) -> Result<Self, FrameiruError> {
        let mut builder = session_builder(&config)?;
        let session = builder.commit_from_memory(bytes).map_err(|e| {
            FrameiruError::Segmentation(format!("failed to load model from memory: {e}"))
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

        let states = [None, None, None, None];
        let zero_states = [
            (vec![1, 16, 1, 1], vec![0f32; 16]),
            (vec![1, 20, 1, 1], vec![0f32; 20]),
            (vec![1, 40, 1, 1], vec![0f32; 40]),
            (vec![1, 64, 1, 1], vec![0f32; 64]),
        ];
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
            zero_states,
            tensor_buf: Vec::new(),
        })
    }

    /// Debug/stateful-testing hook: checksum of the current hidden states.
    pub fn state_checksum(&self) -> u64 {
        self.states
            .iter()
            .flat_map(|s| s.as_ref().map(|(_, data)| data.as_slice()).unwrap_or(&[]))
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

        self.tensor_buf.resize(3 * area, 0.0);
        let letterbox = preprocess_rgb8(
            &frame.data,
            frame.metadata.resolution,
            self.config.input_size,
            self.config.normalization,
            &mut self.tensor_buf,
        )?;

        let src = ort::value::TensorRef::<f32>::from_array_view((
            [1usize, 3, ih, iw],
            self.tensor_buf.as_slice(),
        ))
        .map_err(|e| FrameiruError::Segmentation(format!("tensor build failed: {e}")))?;

        let s0_view = match &self.states[0] {
            Some((shape, data)) => (shape.as_slice(), data.as_slice()),
            None => (
                self.zero_states[0].0.as_slice(),
                self.zero_states[0].1.as_slice(),
            ),
        };
        let s0 = ort::value::TensorRef::<f32>::from_array_view(s0_view)
            .map_err(|e| FrameiruError::Segmentation(format!("state tensor failed: {e}")))?;

        let s1_view = match &self.states[1] {
            Some((shape, data)) => (shape.as_slice(), data.as_slice()),
            None => (
                self.zero_states[1].0.as_slice(),
                self.zero_states[1].1.as_slice(),
            ),
        };
        let s1 = ort::value::TensorRef::<f32>::from_array_view(s1_view)
            .map_err(|e| FrameiruError::Segmentation(format!("state tensor failed: {e}")))?;

        let s2_view = match &self.states[2] {
            Some((shape, data)) => (shape.as_slice(), data.as_slice()),
            None => (
                self.zero_states[2].0.as_slice(),
                self.zero_states[2].1.as_slice(),
            ),
        };
        let s2 = ort::value::TensorRef::<f32>::from_array_view(s2_view)
            .map_err(|e| FrameiruError::Segmentation(format!("state tensor failed: {e}")))?;

        let s3_view = match &self.states[3] {
            Some((shape, data)) => (shape.as_slice(), data.as_slice()),
            None => (
                self.zero_states[3].0.as_slice(),
                self.zero_states[3].1.as_slice(),
            ),
        };
        let s3 = ort::value::TensorRef::<f32>::from_array_view(s3_view)
            .map_err(|e| FrameiruError::Segmentation(format!("state tensor failed: {e}")))?;

        let in_name = self.input_name.as_str();
        let s0_name = self.state_inputs[0].as_str();
        let s1_name = self.state_inputs[1].as_str();
        let s2_name = self.state_inputs[2].as_str();
        let s3_name = self.state_inputs[3].as_str();

        let outputs = match &self.ratio_input {
            Some(ratio_name) => {
                let ratio =
                    ort::value::TensorRef::<f32>::from_array_view(([1usize], &[1.0f32][..]))
                        .map_err(|e| {
                            FrameiruError::Segmentation(format!("ratio tensor failed: {e}"))
                        })?;
                self.session
                    .run(ort::inputs! {
                        in_name => src,
                        s0_name => s0,
                        s1_name => s1,
                        s2_name => s2,
                        s3_name => s3,
                        ratio_name.as_str() => ratio,
                    })
                    .map_err(|e| FrameiruError::Segmentation(format!("inference failed: {e}")))?
            }
            None => self
                .session
                .run(ort::inputs! {
                    in_name => src,
                    s0_name => s0,
                    s1_name => s1,
                    s2_name => s2,
                    s3_name => s3,
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
            let (shape, state_data) = state_ref.extract_tensor();
            let shape_vec: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
            self.states[slot] = Some((shape_vec, state_data.to_vec()));
        }

        let mut out = Mask::filled(frame.metadata.resolution, 0.0);
        if self.config.refine_mask {
            postprocess_mask_refined(
                &model_mask,
                frame.metadata.resolution,
                &letterbox,
                &frame.data,
                self.config.guided_radius,
                self.config.guided_eps,
                &mut out,
            )?;
        } else {
            postprocess_mask(&model_mask, frame.metadata.resolution, &letterbox, &mut out)?;
        }
        crate::polish::polish_mask(&mut out, self.config.mask_dilate, self.config.mask_contrast)?;
        Ok(out)
    }

    fn reset_state(&mut self) {
        self.states = [None, None, None, None];
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

/// Loads the embedded default model (RVM MobileNetV3, 256x256) with
/// `config` — zero-setup segmentation; the model ships inside the binary.
pub fn load_embedded(config: OnnxConfig) -> Result<Box<dyn Segmenter>, FrameiruError> {
    let mut fusion = crate::fusion::FusionSegmenter::load_embedded(config.intra_threads)?;
    fusion.set_roi_zoom(config.roi_zoom);
    Ok(Box::new(fusion))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_linux_cpu_lists() {
        assert_eq!(
            parse_cpu_list("0-3,7,10-12"),
            vec![0, 1, 2, 3, 7, 10, 11, 12]
        );
        assert_eq!(parse_cpu_list("9"), vec![9]);
        assert_eq!(parse_cpu_list("0-1"), vec![0, 1]);
        // Step syntax and garbage are skipped, not fatal.
        assert_eq!(parse_cpu_list("0-7:2,8"), vec![8]);
        assert_eq!(parse_cpu_list(" "), Vec::<u32>::new());
    }

    #[test]
    fn physical_core_count_is_sane() {
        let n = physical_core_count();
        assert!(n >= 1, "physical core count must be >= 1, got {n}");
        // sysfs (when present) must not report more cores than CPUs exist.
        #[cfg(target_os = "linux")]
        if let Some(phys) = sysfs_physical_cores() {
            let logical = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1);
            assert!(phys <= logical, "{phys} physical > {logical} logical");
        }
    }

    #[test]
    fn config_defaults_to_physical_threads() {
        let config = OnnxConfig::new(Resolution {
            width: 256,
            height: 256,
        })
        .unwrap();
        assert_eq!(config.intra_threads, None);
        assert!(config.refine_mask);
        assert_eq!(config.mask_dilate, 0);
        assert_eq!(config.guided_eps, DEFAULT_GUIDED_EPS);
        assert!(!config.roi_zoom);
    }
}
