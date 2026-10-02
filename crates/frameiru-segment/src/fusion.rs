//! Fusion Segmenter: MediaPipe Anchor ∪ RVM Soft Matting.
//!
//! Combines semantic segmentation (MediaPipe) and video matting (RVM):
//!
//! 1. **Semantic Anchor (MediaPipe)**: MediaPipe's face/body detector provides
//!    a confident foreground anchor. Inside the face and torso, the anchor locks
//!    alpha to 1.0, guaranteeing the face and core body are never blurred.
//! 2. **Sub-pixel Hair & Silhouette Matting (RVM)**: RVM provides smooth,
//!    feathered alpha transitions around hair and silhouette edges.

use frameiru_core::buffer::Mask;
use frameiru_core::error::FrameiruError;
#[cfg(any(feature = "onnx", test))]
use frameiru_core::format::Resolution;
#[cfg(feature = "onnx")]
use frameiru_core::traits::Segmenter;
#[cfg(feature = "onnx")]
use frameiru_core::FrameBuffer;

#[cfg(feature = "onnx")]
use crate::model::{
    OnnxConfig, OnnxSegmenter, RvmSegmenter, MEDIAPIPE_MODEL_BYTES, RVM_MODEL_BYTES,
};
#[cfg(feature = "onnx")]
use crate::preprocess::Normalization;
#[cfg(feature = "onnx")]
use crate::roi::{crop_frame, paste_mask_roi, RoiRect, RoiTracker};

/// Fusion segmenter executing MediaPipe as a semantic anchor and RVM as a soft matting engine.
#[cfg(feature = "onnx")]
pub struct FusionSegmenter {
    mediapipe: OnnxSegmenter,
    rvm: RvmSegmenter,
    roi_tracker: RoiTracker,
    roi_zoom_enabled: bool,
}

#[cfg(feature = "onnx")]
impl FusionSegmenter {
    /// Loads the embedded Fusion segmenter (MediaPipe + RVM MobileNetV3).
    pub fn load_embedded(intra_threads: Option<usize>) -> Result<Self, FrameiruError> {
        let mp_config = OnnxConfig {
            input_size: Resolution {
                width: 256,
                height: 256,
            },
            normalization: Normalization::unit(),
            input_name: "input".into(),
            output_name: "output".into(),
            refine_mask: false,
            guided_radius: crate::guided_filter::DEFAULT_GUIDED_RADIUS,
            guided_eps: crate::guided_filter::DEFAULT_GUIDED_EPS,
            intra_threads,
            mask_dilate: 0,
            mask_contrast: 0.0,
            roi_zoom: false,
        };

        let rvm_config = OnnxConfig {
            input_size: Resolution {
                width: 256,
                height: 256,
            },
            normalization: Normalization::default(),
            input_name: "src".into(),
            output_name: "pha".into(),
            refine_mask: false,
            guided_radius: crate::guided_filter::DEFAULT_GUIDED_RADIUS,
            guided_eps: crate::guided_filter::DEFAULT_GUIDED_EPS,
            intra_threads,
            mask_dilate: 0,
            mask_contrast: 0.0,
            roi_zoom: false,
        };

        let mediapipe = OnnxSegmenter::load_bytes(MEDIAPIPE_MODEL_BYTES, mp_config)?;
        let rvm = RvmSegmenter::load_bytes(RVM_MODEL_BYTES, rvm_config)?;

        Ok(Self {
            mediapipe,
            rvm,
            roi_tracker: RoiTracker::new(),
            roi_zoom_enabled: false,
        })
    }

    /// Enables or disables Dynamic Crop & Track (ROI Zoom).
    pub fn set_roi_zoom(&mut self, enabled: bool) {
        self.roi_zoom_enabled = enabled;
        if !enabled {
            self.roi_tracker.reset();
        }
    }

    /// Whether Dynamic Crop & Track (ROI Zoom) is active.
    pub fn roi_zoom(&self) -> bool {
        self.roi_zoom_enabled
    }

    /// Current active ROI crop rectangle, if tracking.
    pub fn current_roi(&self, res: Resolution) -> Option<RoiRect> {
        self.roi_tracker.current_rect(res)
    }
}

#[cfg(feature = "onnx")]
impl Segmenter for FusionSegmenter {
    fn input_resolution(&self) -> Resolution {
        Resolution {
            width: 256,
            height: 256,
        }
    }

    fn segment(&mut self, frame: &FrameBuffer) -> Result<Mask, FrameiruError> {
        let frame_res = frame.metadata.resolution;

        // Dynamic Crop & Track (ROI Zoom) for MediaPipe semantic anchor:
        // Focuses the 256x256 model canvas on the user instead of downscaling the entire room.
        let (mp_res, rvm_res) = if self.roi_zoom_enabled {
            let roi_opt = self.roi_tracker.current_rect(frame_res);
            std::thread::scope(|s| {
                let mp_worker = s.spawn(|| match roi_opt {
                    Some(roi) => {
                        let cropped_frame = crop_frame(frame, roi);
                        let cropped_mask = self.mediapipe.segment(&cropped_frame)?;
                        let mut full_mp = Mask::filled(frame_res, 0.0);
                        paste_mask_roi(&mut full_mp, &cropped_mask, roi);
                        Ok(full_mp)
                    }
                    None => self.mediapipe.segment(frame),
                });
                let rvm_res = self.rvm.segment(frame);
                (
                    mp_worker.join().expect("mediapipe thread panicked"),
                    rvm_res,
                )
            })
        } else {
            std::thread::scope(|s| {
                let mp_worker = s.spawn(|| self.mediapipe.segment(frame));
                let rvm_res = self.rvm.segment(frame);
                (
                    mp_worker.join().expect("mediapipe thread panicked"),
                    rvm_res,
                )
            })
        };

        let mp_mask = mp_res?;
        let rvm_mask = rvm_res?;

        if self.roi_zoom_enabled {
            self.roi_tracker.update_from_mask(&mp_mask);
        }

        let mut fused = Mask::default();
        fuse_masks(&mp_mask, &rvm_mask, &mut fused)?;
        Ok(fused)
    }

    fn reset_state(&mut self) {
        self.mediapipe.reset_state();
        self.rvm.reset_state();
        self.roi_tracker.reset();
    }
}

/// Fuses MediaPipe semantic probability mask with RVM matting alpha.
///
/// - `mp_mask`: Anchor mask from MediaPipe (upsampled to frame resolution).
/// - `rvm_mask`: Matting mask from RVM (upsampled to frame resolution).
/// - `out`: Output destination mask.
pub fn fuse_masks(mp_mask: &Mask, rvm_mask: &Mask, out: &mut Mask) -> Result<(), FrameiruError> {
    if mp_mask.resolution != rvm_mask.resolution {
        return Err(FrameiruError::InvalidArgument(format!(
            "fusion mask resolution mismatch: mp={:?}, rvm={:?}",
            mp_mask.resolution, rvm_mask.resolution
        )));
    }
    let res = mp_mask.resolution;
    let (w, h) = (res.width as usize, res.height as usize);
    let area = w * h;
    if mp_mask.data.len() < area || rvm_mask.data.len() < area {
        return Err(FrameiruError::InvalidArgument(
            "mask buffer smaller than resolution area".into(),
        ));
    }

    out.resolution = res;
    out.data.resize(area, 0.0);

    let mp_slice = &mp_mask.data[..area];
    let rvm_slice = &rvm_mask.data[..area];
    let out_slice = &mut out.data[..area];

    const INV_030: f32 = 1.0 / 0.30;
    const INV_018: f32 = 1.0 / 0.18;
    const INV_095: f32 = 1.0 / 0.95;

    for i in 0..area {
        let m = mp_slice[i];
        let rvm = rvm_slice[i];

        // 1. Core anchor: MediaPipe confident foreground (smoothstep 0.40 -> 0.70)
        let t = ((m - 0.40) * INV_030).clamp(0.0, 1.0);
        let anchor = t * t * (3.0 - 2.0 * t);

        // 2. Semantic Gating: RVM is only allowed within the envelope of the human subject (m >= 0.02).
        // In the distant background where MediaPipe sees no human (m < 0.02), RVM is strictly zeroed out.
        // This eliminates all background ghosting and false positives on room furniture/walls.
        let gate = ((m - 0.02) * INV_018).clamp(0.0, 1.0);
        let rvm_clean = if rvm < 0.05 {
            0.0
        } else {
            (rvm - 0.05) * INV_095
        };
        let rvm_gated = rvm_clean * gate;

        // 3. Fused matte: face/core are locked to anchor (>= 1.0), perimeter uses RVM hair matting
        out_slice[i] = anchor.max(rvm_gated).clamp(0.0, 1.0);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fuse_anchors_face_and_preserves_hair() {
        let res = Resolution {
            width: 100,
            height: 100,
        };
        let mut mp = Mask::filled(res, 0.0);
        let mut rvm = Mask::filled(res, 0.0);

        // Simulate face region: MediaPipe is 0.95, RVM hesitated at 0.20
        let face_idx = 50 * 100 + 50;
        mp.data[face_idx] = 0.95;
        rvm.data[face_idx] = 0.20;

        // Simulate hair strand region: MediaPipe is 0.20, RVM detects hair at 0.75
        let hair_idx = 49 * 100 + 50;
        mp.data[hair_idx] = 0.20;
        rvm.data[hair_idx] = 0.75;

        // Simulate distant background with RVM false positive: MediaPipe is 0.0, RVM hallucinated 0.30
        let bg_idx = 10 * 100 + 10;
        mp.data[bg_idx] = 0.0;
        rvm.data[bg_idx] = 0.30;

        let mut out = Mask::default();
        fuse_masks(&mp, &rvm, &mut out).expect("fusion succeeds");

        // 1. Face must be locked to 1.0 (never blurred)
        assert!(
            out.data[face_idx] > 0.98,
            "face anchor must keep face sharp: got {}",
            out.data[face_idx]
        );

        // 2. Hair strand must preserve RVM soft alpha
        assert!(
            (out.data[hair_idx] - 0.73).abs() < 0.02,
            "hair alpha must be preserved: got {}",
            out.data[hair_idx]
        );

        // 3. Distant background with RVM false positive must be strictly 0.0 (no ghosting)
        assert_eq!(
            out.data[bg_idx], 0.0,
            "background ghosting must be eliminated: got {}",
            out.data[bg_idx]
        );
    }
}
