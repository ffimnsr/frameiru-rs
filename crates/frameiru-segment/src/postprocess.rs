//! Postprocessing: unletterbox + bilinear upsample of model masks.
//!
//! Model output masks live on the letterboxed canvas; the padding region is
//! inference garbage and must be cropped before the mask is upsampled back
//! to the source frame resolution.

use frameiru_core::buffer::Mask;
use frameiru_core::error::FrameiruError;
use frameiru_core::format::Resolution;

use crate::guided_filter::{
    guided_filter, luma_block_avg_rgb8, refine_mask_rgb8, upsample_bilinear,
};
use crate::preprocess::Letterbox;

/// Crops the letterbox padding from `model_mask` and bilinearly upsamples
/// the inner region to `frame_res`, writing raw `f32` values into `out`.
///
/// `model_mask.resolution` must equal the letterbox target. `out` is
/// resized to `frame_res` and overwritten.
fn crop_and_upsample(
    model_mask: &Mask,
    frame_res: Resolution,
    letterbox: &Letterbox,
    out: &mut Vec<f32>,
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

    out.clear();
    out.resize(fx * fy, 0.0);

    let (ox, oy) = (letterbox.pad_x as usize, letterbox.pad_y as usize);
    let ox_f = ox as f32;
    let oy_f = oy as f32;
    let inv_fx = iw as f32 / fx as f32;
    let inv_fy = ih as f32 / fy as f32;

    let mut x_table = Vec::with_capacity(fx);
    for x in 0..fx {
        let s = (x as f32 + 0.5) * inv_fx - 0.5;
        let sx = ox_f + s.clamp(0.0, (iw.saturating_sub(1)) as f32);
        let x0 = sx.floor().min((mw.saturating_sub(1)) as f32) as usize;
        let x1 = (x0 + 1).min(mw.saturating_sub(1));
        let weight_x = sx - x0 as f32;
        x_table.push((x0, x1, weight_x));
    }

    let src = &model_mask.data;
    let max_y = (src.len() / mw).saturating_sub(1);

    for y in 0..fy {
        let s = (y as f32 + 0.5) * inv_fy - 0.5;
        let sy = oy_f + s.clamp(0.0, (ih.saturating_sub(1)) as f32);
        let y0 = (sy.floor() as usize).min(max_y);
        let y1 = (y0 + 1).min(max_y);
        let weight_y = sy - y0 as f32;

        let row0 = &src[y0 * mw..];
        let row1 = &src[y1 * mw..];
        let out_row = &mut out[y * fx..(y + 1) * fx];

        for (out_val, &(x0, x1, weight_x)) in out_row.iter_mut().zip(&x_table) {
            let p00 = row0[x0];
            let p10 = row0[x1];
            let p01 = row1[x0];
            let p11 = row1[x1];

            let top = p00 + (p10 - p00) * weight_x;
            let bottom = p01 + (p11 - p01) * weight_x;
            *out_val = top + (bottom - top) * weight_y;
        }
    }
    Ok(())
}

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
    out.resolution = frame_res;
    crop_and_upsample(model_mask, frame_res, letterbox, &mut out.data)
}

/// Like [`postprocess_mask`], then refines the matte with a guided filter
/// guided by the full-res frame (packed RGB, `frame_res` sized): soft mask
/// edges snap to sharp RGB boundaries (U9.4).
///
/// Frames above 640x480 take the half-resolution path: refine at half res,
/// then bilinear the refined mask back up. Cost drops ~4x at 1080p with
/// sub-pixel edge placement loss (the matte is upsampled from a coarse
/// model mask anyway).
pub fn postprocess_mask_refined(
    model_mask: &Mask,
    frame_res: Resolution,
    letterbox: &Letterbox,
    frame_rgb: &[u8],
    radius: u32,
    eps: f32,
    out: &mut Mask,
) -> Result<(), FrameiruError> {
    let (fx, fy) = (frame_res.width as usize, frame_res.height as usize);
    // 640x480 exact keeps the full-res path (existing behavior/quality);
    // anything larger halves the refine resolution first.
    let large = (fx * fy) > (640 * 480);
    let (tw, th) = if large {
        (fx.div_ceil(2), fy.div_ceil(2))
    } else {
        (fx, fy)
    };
    let target = Resolution {
        width: tw as u32,
        height: th as u32,
    };
    out.resolution = frame_res;
    crop_and_upsample(model_mask, target, letterbox, &mut out.data)?;
    if large {
        let luma = luma_block_avg_rgb8(frame_rgb, fx, fy, 2)?;
        let q = guided_filter(&luma, &out.data, tw, th, radius, eps)?;
        out.data = upsample_bilinear(&q, tw, th, fx, fy);
    } else {
        out.data = refine_mask_rgb8(frame_rgb, &out.data, fx, fy, radius, eps)?;
    }
    Ok(())
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

    /// Top-heavy mask must stay top-heavy through the unletterbox/upsample
    /// path: a flip here would make the compositor blur the wrong half.
    #[test]
    fn postprocess_preserves_vertical_orientation() {
        // Source 4x2 -> target 4x4: scale 1, inner 4x2 at pad_y = 1. The
        // model canvas holds [bg, fg, bg, bg] top-to-bottom; the inner
        // region (mask rows 1..2) must land on frame rows 0..1 in order.
        let lb = Letterbox::compute(res(4, 2), res(4, 4)).unwrap();
        assert_eq!(lb.inner, res(4, 2));
        let model = Mask {
            resolution: res(4, 4),
            data: vec![
                0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        };
        let mut out = Mask::default();
        postprocess_mask(&model, res(4, 2), &lb, &mut out).unwrap();
        assert_eq!(&out.data[..4], &[1.0; 4], "frame top = mask mid row");
        assert_eq!(&out.data[4..], &[0.0; 4], "frame bottom = mask lower row");
    }

    /// Linear-interpolated x where `row` crosses `level`, scanning left→right.
    fn crossing(row: &[f32], level: f32) -> f32 {
        for i in 1..row.len() {
            let (a, b) = (row[i - 1], row[i]);
            if (a > level) != (b > level) {
                return i as f32 - 1.0 + (level - a) / (b - a);
            }
        }
        f32::NAN
    }

    /// Frame where luma at col < split is `lo` and >= split is `hi`.
    fn gray_frame(w: usize, h: usize, lo: u8, hi: u8, split: usize) -> Vec<u8> {
        let mut f = Vec::with_capacity(w * h * 3);
        for _y in 0..h {
            for x in 0..w {
                let v = if x < split { lo } else { hi };
                f.extend_from_slice(&[v, v, v]);
            }
        }
        f
    }

    /// U9.4 regression: through the full postprocess path, the refined mask
    /// edge must follow a sharp RGB boundary better than plain bilinear.
    #[test]
    fn refined_mask_edge_follows_guidance_boundary() {
        // 4x4 low-res mask with a soft edge at model x = 1.67 (cols
        // [0.8, 0.65, 0.5, 0.35]); upscale 2x -> 8x8 frame (no padding).
        let model = Mask {
            resolution: res(4, 4),
            data: (0..4).flat_map(|_| [0.8, 0.65, 0.5, 0.35]).collect(),
        };
        let lb = Letterbox::compute(res(8, 8), res(4, 4)).unwrap();
        assert_eq!(lb.scale, 0.5);

        // Frame: sharp RGB step at x = 3.5 (lo=0.2, hi=0.8 split col 4).
        let frame = gray_frame(8, 8, 51, 204, 4);

        let mut plain = Mask::default();
        postprocess_mask(&model, res(8, 8), &lb, &mut plain).unwrap();
        let mut refined = Mask::default();
        postprocess_mask_refined(&model, res(8, 8), &lb, &frame, 2, 1e-3, &mut refined).unwrap();

        let mid = 4; // middle row, away from box-filter borders
        let p_cross = crossing(&plain.data[mid * 8..mid * 8 + 8], 0.5);
        let q_cross = crossing(&refined.data[mid * 8..mid * 8 + 8], 0.5);
        // Bilinear p crosses at x = 4.5; the true RGB edge is at x = 3.5.
        assert!((p_cross - 4.5).abs() < 1e-3, "plain crosses at {p_cross}");
        assert!(
            q_cross < p_cross - 0.2,
            "refined edge ({q_cross}) must move toward the RGB edge (3.5), plain was {p_cross}"
        );
        // And stay roughly on the RGB boundary, not overshoot.
        assert!(
            (q_cross - 3.5).abs() < 1.0,
            "refined edge {q_cross} far from RGB edge"
        );
        assert!(
            (q_cross - 3.5).abs() < 1.0,
            "refined edge {q_cross} far from RGB edge"
        );
        // Far from the edge the mask must stay put (no texture bleed).
        for x in [0, 1, 6, 7] {
            assert!(
                (refined.data[mid * 8 + x] - plain.data[mid * 8 + x]).abs() < 0.15,
                "x={x}: plain {} refined {}",
                plain.data[mid * 8 + x],
                refined.data[mid * 8 + x]
            );
        }
    }

    #[test]
    fn refined_identity_letterbox_preserves_mask() {
        // Radius 0 (1x1 window): cov/var vanish exactly, so q == p
        // regardless of guidance.
        let model = Mask {
            resolution: res(2, 2),
            data: vec![0.1, 0.5, 0.9, 0.3],
        };
        let lb = Letterbox::compute(res(2, 2), res(2, 2)).unwrap();
        let frame = vec![128u8; 2 * 2 * 3];
        let mut out = Mask::default();
        postprocess_mask_refined(&model, res(2, 2), &lb, &frame, 0, 1e-3, &mut out).unwrap();
        for i in 0..4 {
            assert!(
                (out.data[i] - model.data[i]).abs() < 1e-5,
                "refined[{i}] {} != {}",
                out.data[i],
                model.data[i]
            );
        }
    }

    #[test]
    fn refined_rejects_short_frame_data() {
        let model = mask_at(res(4, 4), 0.5);
        let lb = Letterbox::compute(res(4, 4), res(4, 4)).unwrap();
        let mut out = Mask::default();
        assert!(
            postprocess_mask_refined(&model, res(4, 4), &lb, &[0u8; 9], 2, 1e-3, &mut out).is_err()
        );
    }

    /// U9.6: frames above 640x480 refine at half resolution; the edge must
    /// still follow a sharp RGB boundary (within a couple pixels) and output
    /// at the full frame resolution. Geometry mimics production: a 256-wide
    /// model mask (like the landscape mediapipe) upscaled 5x to 1280x720,
    /// with the luma boundary 4 px right of the mask edge (inside the
    /// filter window).
    #[test]
    fn refined_high_res_uses_half_path_and_follows_edges() {
        let fw = 1280u32;
        let fh = 720u32;
        let mw = 256u32;
        // Soft vertical edge crossing 0.5 at model col 128 -> full x = 640.
        let model = Mask {
            resolution: res(mw, 144),
            data: (0..144)
                .flat_map(|_| (0..256).map(|x| (0.8 - (x - 126) as f32 * 0.15).clamp(0.1, 0.8)))
                .collect(),
        };
        let lb = Letterbox::compute(res(fw, fh), res(mw, 144)).unwrap();
        assert_eq!(lb.scale, 0.2); // 5x upscale on the way back out
                                   // Frame with a sharp luma step at x = 644 (left dark 0.25, right 0.75).
        let mut frame = vec![0u8; (fw * fh * 3) as usize];
        for y in 0..fh {
            for x in 0..fw {
                let v = if x < 644 { 64u8 } else { 191u8 };
                let i = (y * fw + x) as usize * 3;
                frame[i] = v;
                frame[i + 1] = v;
                frame[i + 2] = v;
            }
        }
        let mut plain = Mask::default();
        postprocess_mask(&model, res(fw, fh), &lb, &mut plain).unwrap();
        let mut refined = Mask::default();
        postprocess_mask_refined(&model, res(fw, fh), &lb, &frame, 8, 1e-3, &mut refined).unwrap();

        assert_eq!(refined.resolution, res(fw, fh));
        assert_eq!(refined.data.len(), (fw * fh) as usize);
        let mid = (fh / 2) as usize;
        let mut p_row = [0f32; 1280];
        let mut q_row = [0f32; 1280];
        p_row.copy_from_slice(&plain.data[mid * fw as usize..(mid + 1) * fw as usize]);
        q_row.copy_from_slice(&refined.data[mid * fw as usize..(mid + 1) * fw as usize]);
        let p_cross = crossing(&p_row, 0.5);
        let q_cross = crossing(&q_row, 0.5);
        assert!(
            (p_cross - 642.0).abs() < 0.1,
            "test geometry: plain edge should sit at ~642 (texel-center), got {p_cross}"
        );
        assert!(
            (q_cross - 644.0).abs() < (p_cross - 644.0).abs(),
            "refined edge {q_cross} must move toward the luma boundary (644), plain was {p_cross}"
        );
        assert!(
            (q_cross - 644.0).abs() < 2.0,
            "refined edge {q_cross} should land within 2px of 644"
        );
        // Values stay in the mask range.
        assert!(refined.data.iter().all(|&v| (0.0..=1.0).contains(&v)));
    }
}
