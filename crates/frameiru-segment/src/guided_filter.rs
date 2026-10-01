//! Guided filter (He et al., "Guided Image Filtering", ECCV 2010) for
//! edge-aware mask upsampling.
//!
//! Low-res model masks lose hair/contour edges when upscaled; the guided
//! filter re-aligns the soft mask edge to the full-res frame's edges using
//! luma as guidance, at O(N) cost (separable box filters) instead of a
//! brute-force joint bilateral filter.
//!
//! Output `q = mean(a)·I + mean(b)` with `a,b` solved per local window:
//! flat mask regions (`cov ≈ 0`) are smoothed to the local mean — kills
//! blockiness without bleeding frame texture into the matte; regions where
//! mask and guidance edges correlate snap to the guidance edge.

use frameiru_core::error::FrameiruError;

/// Default guided-filter neighborhood radius (box half-size, in pixels).
pub const DEFAULT_GUIDED_RADIUS: u32 = 8;
/// Default regularization: luma variance below `eps` is treated as flat.
/// Masks/luma live in [0, 1], so `1e-3` keeps strong edges while smoothing
/// low-contrast regions.
pub const DEFAULT_GUIDED_EPS: f32 = 0.02;

/// Separable box-filter mean with clamp (replicate-edge) padding, O(N) in the
/// radius: windows are always the full `(2r+1)` span with edge values
/// replicated, so flat regions stay flat up to the borders.
pub(crate) fn box_filter(src: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    let span = (2 * r + 1) as f64;
    let inv_span = 1.0 / span;
    let n = w * h;
    let mut tmp = vec![0f32; n];
    for y in 0..h {
        let base = y * w;
        let mut acc = 0.0f64;
        // Position 0: clamped positions -r..=0 are all src[..0] (r+1 copies),
        // then 1..=r each clamp to min(i, w-1).
        acc += (r + 1) as f64 * src[base] as f64;
        for i in 1..=r {
            acc += src[base + i.min(w - 1)] as f64;
        }
        tmp[base] = (acc * inv_span) as f32;
        for x in 1..w {
            // Slide: drop clamped position x-1-r, add clamped position x+r.
            acc -= src[base + (x - 1).saturating_sub(r)] as f64;
            acc += src[base + (x + r).min(w - 1)] as f64;
            tmp[base + x] = (acc * inv_span) as f32;
        }
    }
    let mut out = vec![0f32; n];
    for x in 0..w {
        let mut acc = 0.0f64;
        acc += (r + 1) as f64 * tmp[x] as f64;
        for y in 1..=r {
            acc += tmp[y.min(h - 1) * w + x] as f64;
        }
        out[x] = (acc * inv_span) as f32;
        for y in 1..h {
            acc -= tmp[(y - 1).saturating_sub(r) * w + x] as f64;
            acc += tmp[(y + r).min(h - 1) * w + x] as f64;
            out[y * w + x] = (acc * inv_span) as f32;
        }
    }
    out
}

/// Guided filter of filter input `p` (the upsampled mask) along guidance `I`
/// (frame luma in [0, 1]), both `w×h`. Returns the refined `q`.
pub fn guided_filter(
    guidance: &[f32],
    p: &[f32],
    w: usize,
    h: usize,
    radius: u32,
    eps: f32,
) -> Result<Vec<f32>, FrameiruError> {
    if w == 0 || h == 0 {
        return Err(FrameiruError::InvalidArgument(
            "guided filter requires non-zero dimensions".into(),
        ));
    }
    let n = w * h;
    if guidance.len() < n || p.len() < n {
        return Err(FrameiruError::InsufficientCapacity {
            need: n,
            have: guidance.len().min(p.len()),
        });
    }
    if !eps.is_finite() || eps < 0.0 {
        return Err(FrameiruError::InvalidArgument(format!(
            "guided filter eps must be finite and >= 0, got {eps}"
        )));
    }

    // Radius 0 gives a 1x1 window: cov/var vanish exactly, so q == p.
    let r = (radius as usize).min(w.max(h));

    // Fast guided filter (He's subsampling): solve a/b on a 4x4-downsampled
    // lattice, bilinearly upsample mean(a)/mean(b), then apply at full res.
    // Boxes then cost 1/16 of the full-res filter; guidance statistic
    // windows keep their full-res extent. Small images keep the exact path.
    let subsample = w >= 64 && h >= 64 && r >= 4;
    let output = if subsample {
        let (ws, hs) = (w / 4, h / 4);
        let (gs, ps) = downsample4(guidance, p, w, h);
        let rs = (r / 4).max(1);
        let (ma, mb) = solve_ab(&gs, &ps, ws, hs, rs, eps);
        let ma = upsample_bilinear(&ma, ws, hs, w, h);
        let mb = upsample_bilinear(&mb, ws, hs, w, h);
        combine(&ma, &mb, guidance)
    } else {
        let (ma, mb) = solve_ab(guidance, p, w, h, r, eps);
        combine(&ma, &mb, guidance)
    };
    Ok(output)
}

/// Solves the guided-filter linear model `q = mean(a)·I + mean(b)` for a
/// single resolution: box statistics, then `a,b` per pixel, then box `a,b`.
fn solve_ab(
    guidance: &[f32],
    p: &[f32],
    w: usize,
    h: usize,
    r: usize,
    eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    let n = w * h;
    let mean_i = box_filter(guidance, w, h, r);
    let mean_p = box_filter(p, w, h, r);

    let mut corr_i = vec![0f32; n];
    let mut corr_ip = vec![0f32; n];
    for i in 0..n {
        let g = guidance[i];
        corr_i[i] = g * g;
        corr_ip[i] = g * p[i];
    }
    let corr_i = box_filter(&corr_i, w, h, r);
    let corr_ip = box_filter(&corr_ip, w, h, r);

    let mut a = vec![0f32; n];
    let mut b = vec![0f32; n];
    for i in 0..n {
        let var = corr_i[i] - mean_i[i] * mean_i[i];
        let cov = corr_ip[i] - mean_i[i] * mean_p[i];
        a[i] = cov / (var + eps);
        b[i] = mean_p[i] - a[i] * mean_i[i];
    }
    (box_filter(&a, w, h, r), box_filter(&b, w, h, r))
}

fn combine(mean_a: &[f32], mean_b: &[f32], guidance: &[f32]) -> Vec<f32> {
    (0..guidance.len())
        .map(|i| mean_a[i] * guidance[i] + mean_b[i])
        .collect()
}

/// 4x4 block-average downsample (clamped on partial edge blocks), matching
/// the guidance-statistics heuristic of the fast guided filter.
fn downsample4(guidance: &[f32], p: &[f32], w: usize, h: usize) -> (Vec<f32>, Vec<f32>) {
    let (ws, hs) = (w / 4, h / 4);
    let mut gs = vec![0f32; ws * hs];
    let mut ps = vec![0f32; ws * hs];
    for ly in 0..hs {
        let y0 = ly * 4;
        let y1 = (y0 + 4).min(h);
        for lx in 0..ws {
            let x0 = lx * 4;
            let x1 = (x0 + 4).min(w);
            let mut sg = 0.0f64;
            let mut sp = 0.0f64;
            let mut count = 0usize;
            for y in y0..y1 {
                for x in x0..x1 {
                    sg += guidance[y * w + x] as f64;
                    sp += p[y * w + x] as f64;
                    count += 1;
                }
            }
            let inv = 1.0 / count as f64;
            let idx = ly * ws + lx;
            gs[idx] = (sg * inv) as f32;
            ps[idx] = (sp * inv) as f32;
        }
    }
    (gs, ps)
}

/// Bilinear upsample `src` (`ws×hs`) to `w×h`, texel-center mapping with
/// edge clamping. Release uses the SIMD `fast_image_resize` (already
/// vendored for preprocessing); debug builds keep a fused scalar sampler
/// (the SIMD paths collapse to slow fallbacks in debug, as in preprocessing).
pub(crate) fn upsample_bilinear(src: &[f32], ws: usize, hs: usize, w: usize, h: usize) -> Vec<f32> {
    if w == ws && h == hs {
        return src.to_vec();
    }
    #[cfg(debug_assertions)]
    {
        let mut out = vec![0f32; w * h];
        for y in 0..h {
            let sy = ((y as f32 + 0.5) * (hs as f32 / h as f32) - 0.5).clamp(0.0, (hs - 1) as f32);
            let y0 = sy.floor() as usize;
            let y1 = (y0 + 1).min(hs - 1);
            let fy = sy - y0 as f32;
            for x in 0..w {
                let sx =
                    ((x as f32 + 0.5) * (ws as f32 / w as f32) - 0.5).clamp(0.0, (ws - 1) as f32);
                let x0 = sx.floor() as usize;
                let x1 = (x0 + 1).min(ws - 1);
                let fx = sx - x0 as f32;
                let p00 = src[y0 * ws + x0];
                let p10 = src[y0 * ws + x1];
                let p01 = src[y1 * ws + x0];
                let p11 = src[y1 * ws + x1];
                let top = p00 + (p10 - p00) * fx;
                let bottom = p01 + (p11 - p01) * fx;
                out[y * w + x] = top + (bottom - top) * fy;
            }
        }
        out
    }
    #[cfg(not(debug_assertions))]
    {
        // ImageRef buffers are byte slices regardless of pixel type. The
        // source is only read by the resizer, so a with-aligned f32 view can
        // be reinterpreted as bytes soundly (F32 is 4 bytes, len*4 is a
        // multiple of 4).
        // SAFETY: `src` is a well-aligned `f32` slice; viewing it as bytes is
        // always valid and the resizer never mutates its source view.
        let bytes = unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u8, src.len() * 4) };
        let src_view = fast_image_resize::images::ImageRef::new(
            ws as u32,
            hs as u32,
            bytes,
            fast_image_resize::PixelType::F32,
        )
        .expect("upsample source view: valid dims");
        let mut out = fast_image_resize::images::Image::new(
            w as u32,
            h as u32,
            fast_image_resize::PixelType::F32,
        );
        let mut resizer = fast_image_resize::Resizer::new();
        let options = fast_image_resize::ResizeOptions::new().resize_alg(
            fast_image_resize::ResizeAlg::Interpolation(fast_image_resize::FilterType::Bilinear),
        );
        resizer
            .resize(&src_view, &mut out, &options)
            .expect("upsample resize");
        out.into_vec()
            .chunks_exact(4)
            .map(|c| f32::from_ne_bytes(c.try_into().expect("4-byte chunk")))
            .collect()
    }
}

/// BT.601 luma of a packed-RGB frame, normalized to [0, 1].
pub(crate) fn luma_from_rgb8(src: &[u8], w: usize, h: usize) -> Result<Vec<f32>, FrameiruError> {
    let n = w * h;
    let need = n * 3;
    if src.len() < need {
        return Err(FrameiruError::InsufficientCapacity {
            need,
            have: src.len(),
        });
    }
    Ok(src[..need]
        .chunks_exact(3)
        .map(|px| (0.299 * px[0] as f32 + 0.587 * px[1] as f32 + 0.114 * px[2] as f32) / 255.0)
        .collect())
}

/// Block-averaged BT.601 luma: one fused pass over the RGB frame producing
/// `(w/block) x (h/block)` luma cells in [0, 1] (clamped on partial edge
/// blocks). Used by the half-resolution refine path so the full-res frame is
/// scanned once instead of luma-plus-downsample.
pub(crate) fn luma_block_avg_rgb8(
    src: &[u8],
    w: usize,
    h: usize,
    block: usize,
) -> Result<Vec<f32>, FrameiruError> {
    if block == 0 {
        return Err(FrameiruError::InvalidArgument(
            "luma block size must be >= 1".into(),
        ));
    }
    let (cw, ch) = (w.div_ceil(block), h.div_ceil(block));
    let need = w * h * 3;
    if src.len() < need {
        return Err(FrameiruError::InsufficientCapacity {
            need,
            have: src.len(),
        });
    }
    let mut out = vec![0f32; cw * ch];
    for cy in 0..ch {
        let y0 = cy * block;
        let y1 = (y0 + block).min(h);
        for cx in 0..cw {
            let x0 = cx * block;
            let x1 = (x0 + block).min(w);
            let mut acc = 0.0f64;
            let mut count = 0usize;
            for y in y0..y1 {
                let base = y * w;
                for x in x0..x1 {
                    let i = (base + x) * 3;
                    acc += 0.299 * src[i] as f64
                        + 0.587 * src[i + 1] as f64
                        + 0.114 * src[i + 2] as f64;
                    count += 1;
                }
            }
            let inv = 1.0 / count as f64;
            out[cy * cw + cx] = ((acc * inv) / 255.0) as f32;
        }
    }
    Ok(out)
}

/// Refines the upsampled mask `p` with `src` (packed RGB, `w×h`) as guidance.
pub fn refine_mask_rgb8(
    src: &[u8],
    p: &[f32],
    w: usize,
    h: usize,
    radius: u32,
    eps: f32,
) -> Result<Vec<f32>, FrameiruError> {
    let luma = luma_from_rgb8(src, w, h)?;
    guided_filter(&luma, p, w, h, radius, eps)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn res_vec(w: usize, h: usize, fill: impl Fn(usize, usize) -> f32) -> Vec<f32> {
        let mut out = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                out.push(fill(x, y));
            }
        }
        out
    }

    /// Width-aware row view.
    fn row(data: &[f32], w: usize, y: usize) -> &[f32] {
        &data[y * w..(y + 1) * w]
    }

    /// Linear-interpolated x where `row` crosses `level`, scanning left→right.
    fn crossing(row: &[f32], level: f32) -> f32 {
        for i in 1..row.len() {
            let (a, b) = (row[i - 1], row[i]);
            if (a > level) != (b > level) {
                let t = (level - a) / (b - a);
                return i as f32 - 1.0 + t;
            }
        }
        f32::NAN
    }

    /// The 4x subsampled fast path must reproduce the exact filter's edge
    /// behavior on real-time-sized frames (80x80 >= 64 threshold).
    #[test]
    fn fast_path_matches_exact_path_on_real_time_sizes() {
        let (w, h) = (80, 80);
        let guidance = res_vec(w, h, |x, _| if x < 40 { 0.2 } else { 0.8 });
        let p = res_vec(w, h, |x, _| match x {
            0..=34 => 0.8,
            35 => 0.7,
            36 => 0.6,
            37 => 0.5,
            38 => 0.4,
            39 => 0.3,
            _ => 0.2,
        });
        // Exact path at full res.
        let (ma, mb) = solve_ab(&guidance, &p, w, h, 8, 1e-3);
        let exact = combine(&ma, &mb, &guidance);
        // Public API: fast path (subsample 4).
        let fast = guided_filter(&guidance, &p, w, h, 8, 1e-3).unwrap();

        let max_dev = fast
            .iter()
            .zip(&exact)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_dev < 0.1, "fast path diverges from exact by {max_dev}");
        // Both must snap the soft edge (crossing at 37.0) toward the
        // guidance boundary (39.5).
        let fast_cross = crossing(row(&fast, w, 40), 0.5);
        let exact_cross = crossing(row(&exact, w, 40), 0.5);
        assert!(
            fast_cross > 37.5 && exact_cross > 37.5,
            "edges must follow the RGB boundary: fast {fast_cross}, exact {exact_cross}"
        );
        assert!(
            (fast_cross - exact_cross).abs() < 1.0,
            "fast crossing {fast_cross} differs from exact {exact_cross}"
        );
    }

    #[test]
    fn flat_mask_survives_textured_guidance_unchanged() {
        // Constant p must stay constant: cov == 0 everywhere, so a == 0 and
        // q == local mean of p. No guidance texture may bleed into the matte.
        // Clamp padding keeps this exact up to the borders.
        let (w, h) = (16, 16);
        let guidance: Vec<f32> = res_vec(w, h, |x, y| if (x + y) % 2 == 0 { 0.95 } else { 0.05 });
        let p = vec![0.5; w * h];
        let q = guided_filter(&guidance, &p, w, h, 3, 1e-3).unwrap();
        assert!(
            q.iter().all(|&v| (v - 0.5).abs() < 1e-3),
            "flat matte must not pick up guidance texture: {:?}",
            &q[..16]
        );
    }

    #[test]
    fn uniform_guidance_falls_back_to_box_mean() {
        // a == 0 under flat guidance: q == mean_a*I + mean_b == box(b),
        // and b == mean_p, so q is box applied twice.
        let (w, h) = (10, 10);
        let guidance = vec![0.5; w * h];
        let p: Vec<f32> = res_vec(w, h, |x, y| ((x + y) % 5) as f32 / 4.0);
        let q = guided_filter(&guidance, &p, w, h, 2, 1e-3).unwrap();
        let first = box_filter(&p, w, h, 2);
        let expected = box_filter(&first, w, h, 2);
        for i in 0..p.len() {
            assert!(
                (q[i] - expected[i]).abs() < 1e-5,
                "q[{i}] {} != {}",
                q[i],
                expected[i]
            );
        }
    }

    /// U9.4 regression: a soft mask edge must be pulled toward a sharp RGB
    /// boundary that sits within the filter window.
    #[test]
    fn mask_edge_follows_sharp_guidance_boundary() {
        let (w, h) = (16, 16);
        // Sharp guidance step at x = 7.5 (cols 0..7 dark, 8..15 bright).
        let guidance = res_vec(w, h, |x, _| if x < 8 { 0.2 } else { 0.8 });
        // Soft mask step crossing 0.5 at x = 7.0 (cols 0..5 -> 0.8, ramp to
        // 0.2 by col 9): the matte edge is offset 0.5 px from the RGB edge.
        let p = res_vec(w, h, |x, _| match x {
            0..=5 => 0.8,
            6 => 0.65,
            7 => 0.5,
            8 => 0.35,
            _ => 0.2,
        });
        let p_cross = crossing(row(&p, w, 8), 0.5);
        assert!(
            (p_cross - 7.0).abs() < 1e-3,
            "test setup: p crosses at {p_cross}"
        );

        let q = guided_filter(&guidance, &p, w, h, 2, 1e-3).unwrap();
        let q_cross = crossing(row(&q, w, 8), 0.5);
        // Edge must move toward the guidance boundary (7.5) and stay close.
        assert!(
            q_cross > 7.15,
            "mask edge must follow the RGB boundary: p crossed at {p_cross}, q at {q_cross}"
        );
        assert!(
            (q_cross - 7.5).abs() < (p_cross - 7.5).abs(),
            "q edge ({q_cross}) must be closer to the RGB edge (7.5) than p's ({p_cross})"
        );
        // The transition must also sharpen: the q band [0.4, 0.6] is
        // narrower than p's soft ramp.
        let band = |data: &[f32]| -> f32 {
            let r0 = row(data, w, 8);
            (crossing(r0, 0.6) - crossing(r0, 0.4)).abs()
        };
        assert!(
            band(&q) < band(&p),
            "refined transition ({} px) must be sharper than p's ({} px)",
            band(&q),
            band(&p)
        );
        // Flat regions away from the boundary must survive unchanged.
        for x in [1, 2, 3, 13, 14] {
            assert!(
                (q[8 * w + x] - p[8 * w + x]).abs() < 0.1,
                "flat region at x={x} shifted: p={} q={}",
                p[8 * w + x],
                q[8 * w + x]
            );
        }
    }

    #[test]
    fn zero_radius_is_identity() {
        let (w, h) = (8, 8);
        let guidance: Vec<f32> = res_vec(w, h, |x, y| ((x * 7 + y * 3) % 11) as f32 / 10.0);
        let p: Vec<f32> = (0..w * h).map(|i| (i % 7) as f32 / 6.0).collect();
        let q = guided_filter(&guidance, &p, w, h, 0, 1e-3).unwrap();
        for i in 0..p.len() {
            assert!(
                (q[i] - p[i]).abs() < 1e-5,
                "q[{i}] {} != p {} (r=0 must be identity)",
                q[i],
                p[i]
            );
        }
    }

    #[test]
    fn luma_of_packed_rgb_matches_bt601() {
        let w = 4;
        let h = 1;
        // White, black, pure red, pure green.
        let src = [255u8, 255, 255, 0, 0, 0, 255, 0, 0, 0, 255, 0];
        let luma = luma_from_rgb8(&src, w, h).unwrap();
        assert_eq!(luma.len(), 4);
        let expected = [1.0, 0.0, 0.299, 0.587];
        for i in 0..4 {
            assert!(
                (luma[i] - expected[i]).abs() < 1e-4,
                "luma[{i}] = {}",
                luma[i]
            );
        }
    }

    #[test]
    fn validates_inputs() {
        assert!(guided_filter(&[0.5; 4], &[0.5; 4], 2, 2, 2, 1e-3).is_ok());
        // Short p.
        assert!(guided_filter(&[0.5; 16], &[0.5; 4], 4, 4, 2, 1e-3).is_err());
        // Zero dims.
        assert!(guided_filter(&[], &[], 0, 4, 2, 1e-3).is_err());
        // Negative eps.
        assert!(guided_filter(&[0.5; 16], &[0.5; 16], 4, 4, 2, -1.0).is_err());
        // Short RGB source.
        assert!(refine_mask_rgb8(&[0u8; 3], &[0.5; 9], 3, 3, 2, 1e-3).is_err());
    }
}
