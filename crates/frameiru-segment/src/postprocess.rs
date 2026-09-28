//! Postprocessing: unletterbox + bilinear upsample of model masks.
//!
//! Model output masks live on the letterboxed canvas; the padding region is
//! inference garbage and must be cropped before the mask is upsampled back
//! to the source frame resolution.

use frameiru_core::buffer::Mask;
use frameiru_core::error::FrameiruError;
use frameiru_core::format::Resolution;

use crate::preprocess::Letterbox;

/// Crops the letterbox padding from `model_mask` and bilinearly upsamples
/// the inner region to `frame_res`.
///
/// `model_mask.resolution` must equal the letterbox target. `out` is
/// resized to `frame_res` and overwritten.
pub fn postprocess_mask(
    model_mask: &Mask,
    frame_res: Resolution,
    letterbox: &Letterbox,
    out: &mut Mask,
) -> Result<(), FrameiruError> {
    if model_mask.resolution != letterbox.target {
        return Err(FrameiruError::InvalidArgument(format!(
            "model mask resolution {:?} does not match letterbox target {:?}",
            model_mask.resolution, letterbox.target
        )));
    }
    if !frame_res.is_valid() {
        return Err(FrameiruError::InvalidArgument(
            "frame resolution must be non-zero".into(),
        ));
    }

    let (mw, mh) = (
        model_mask.resolution.width as usize,
        model_mask.resolution.height as usize,
    );
    let (iw, ih) = (
        letterbox.inner.width as usize,
        letterbox.inner.height as usize,
    );
    let (fx, fy) = (frame_res.width as usize, frame_res.height as usize);
    let model_area = mw * mh;
    if model_mask.data.len() < model_area {
        return Err(FrameiruError::InsufficientCapacity {
            need: model_area,
            have: model_mask.data.len(),
        });
    }

    out.resolution = frame_res;
    out.data.resize(fx * fy, 0.0);

    let (ox, oy) = (letterbox.pad_x as usize, letterbox.pad_y as usize);
    for y in 0..fy {
        for x in 0..fx {
            // Sample the inner region using the same texel-center mapping as
            // preprocessing, so resize round-trips are stable.
            let sx = sample_coord(x as f32, fx as f32, iw as f32);
            let sy = sample_coord(y as f32, fy as f32, ih as f32);
            out.data[y * fx + x] =
                bilinear_sample_f32(&model_mask.data, mw, ox as f32 + sx, oy as f32 + sy);
        }
    }
    Ok(())
}

/// Maps a destination pixel to a source coordinate (texel-center convention),
/// clamped to the source edge.
fn sample_coord(dst: f32, dst_size: f32, src_size: f32) -> f32 {
    let s = (dst + 0.5) * (src_size / dst_size) - 0.5;
    s.clamp(0.0, src_size - 1.0)
}

/// Bilinear sample of a f32 mask row at `sx`/`sy` (absolute, pre-clamped).
fn bilinear_sample_f32(src: &[f32], row_stride: usize, sx: f32, sy: f32) -> f32 {
    let x0 = sx.floor().min((row_stride - 1) as f32) as usize;
    let y0 = sy.floor() as usize;
    let x1 = (x0 + 1).min(row_stride - 1);
    let y1 = (y0 + 1).min(src.len() / row_stride - 1);
    let fx = sx - x0 as f32;
    let fy = sy - y0 as f32;

    let p00 = src[y0 * row_stride + x0];
    let p10 = src[y0 * row_stride + x1];
    let p01 = src[y1 * row_stride + x0];
    let p11 = src[y1 * row_stride + x1];

    let top = p00 + (p10 - p00) * fx;
    let bottom = p01 + (p11 - p01) * fx;
    top + (bottom - top) * fy
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preprocess::Letterbox;
    use frameiru_core::buffer::Mask;

    fn res(w: u32, h: u32) -> Resolution {
        Resolution {
            width: w,
            height: h,
        }
    }

    fn mask_at(resolution: Resolution, fill: f32) -> Mask {
        let area = resolution.area() as usize;
        Mask {
            resolution,
            data: vec![fill; area],
        }
    }

    #[test]
    fn identity_letterbox_preserves_mask() {
        // Same resolution, no padding: output equals input exactly.
        let model = Mask {
            resolution: res(2, 2),
            data: vec![0.1, 0.5, 0.9, 0.3],
        };
        let lb = Letterbox::compute(res(2, 2), res(2, 2)).unwrap();
        let mut out = Mask::default();
        postprocess_mask(&model, res(2, 2), &lb, &mut out).unwrap();
        assert_eq!(out.resolution, res(2, 2));
        assert_eq!(out.data, vec![0.1, 0.5, 0.9, 0.3]);
    }

    #[test]
    fn constant_mask_stays_constant() {
        let model = mask_at(res(4, 4), 0.75);
        let lb = Letterbox::compute(res(2, 2), res(4, 4)).unwrap();
        let mut out = Mask::default();
        postprocess_mask(&model, res(2, 2), &lb, &mut out).unwrap();
        assert!(out.data.iter().all(|&v| (v - 0.75).abs() < 1e-6));
    }

    #[test]
    fn bilinear_upsample_matches_hand_computed_values() {
        // 2x2 mask [[0,1],[1,0]] upscaled 2x: x coords map to 0, .25, .75, 1.
        let model = Mask {
            resolution: res(2, 2),
            data: vec![0.0, 1.0, 1.0, 0.0],
        };
        let lb = Letterbox::compute(res(4, 4), res(2, 2)).unwrap();
        assert_eq!(lb.scale, 0.5);
        let mut out = Mask::default();
        postprocess_mask(&model, res(4, 4), &lb, &mut out).unwrap();
        // Top row: lerp(0, 1) at 0, .25, .75, 1.
        assert_eq!(out.data[..4], [0.0, 0.25, 0.75, 1.0]);
        // Row 1: vertical lerp at fy = .25 between top and bottom rows.
        let top = [0.0, 0.25, 0.75, 1.0];
        let bottom = [1.0, 0.75, 0.25, 0.0];
        let mid: Vec<f32> = top
            .into_iter()
            .zip(bottom)
            .map(|(t, b)| t + (b - t) * 0.25)
            .collect();
        assert_eq!(out.data[4..8], mid);
    }

    #[test]
    fn padding_region_is_excluded() {
        // Model 4x4 with garbage in padded rows; inner region 4x2 at pad_y 1.
        let mut data = vec![9.0; 16]; // garbage padding
                                      // Inner rows: [0, 1] top, [2, 3] bottom pattern.
        data[4..8].copy_from_slice(&[0.0, 0.25, 0.75, 1.0]);
        data[8..12].copy_from_slice(&[1.0, 0.75, 0.25, 0.0]);
        let model = Mask {
            resolution: res(4, 4),
            data,
        };
        // Source 4x2 -> target 4x4: scale 1, inner 4x2, pad_y 1.
        let lb = Letterbox::compute(res(4, 2), res(4, 4)).unwrap();
        let mut out = Mask::default();
        postprocess_mask(&model, res(4, 2), &lb, &mut out).unwrap();
        assert_eq!(out.resolution, res(4, 2));
        // Only inner content survives; no 9.0 padding values.
        assert!(out.data.iter().all(|&v| v <= 1.0));
        assert_eq!(out.data[..4], [0.0, 0.25, 0.75, 1.0]);
    }

    #[test]
    fn rejects_wrong_model_resolution() {
        let model = mask_at(res(4, 4), 0.5);
        let mut out = Mask::default();
        // Letterbox whose target differs from the model mask resolution.
        let bad = Letterbox::compute(res(2, 2), res(3, 3)).unwrap();
        assert!(postprocess_mask(&model, res(2, 2), &bad, &mut out).is_err());
    }

    #[test]
    fn rejects_undersized_mask_data() {
        let model = Mask {
            resolution: res(8, 8),
            data: vec![0.0; 10],
        };
        let lb = Letterbox::compute(res(4, 4), res(8, 8)).unwrap();
        let mut out = Mask::default();
        assert!(postprocess_mask(&model, res(4, 4), &lb, &mut out).is_err());
    }
}
