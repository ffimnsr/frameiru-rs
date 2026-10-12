//! Extended CPU overlay effects (GPU feature parity in `shaders.rs`).
//!
//! Kept out of [`cpu`](crate::cpu) so the compositor core stays focused on
//! blending. The GPU shader branches mirror these formulas. Every effect is a
//! deterministic function of `(x, y, t)` so the rayon-parallel row loops
//! cannot race; effects that resample take one frame copy up front.

use frameiru_core::OverlayMode;
use rayon::prelude::*;

/// Luma weights shared with the GPU shader branches.
const LUMA: [f32; 3] = [0.299, 0.587, 0.114];

/// Applies one of the extended overlay modes to the composited frame.
///
/// Returns `false` for the classic modes handled by
/// [`cpu::apply_overlay`](crate::cpu), so the caller can fall through.
pub(crate) fn apply(out: &mut [u8], w: usize, h: usize, mode: OverlayMode, t: f64) -> bool {
    match mode {
        OverlayMode::ChromaticAberration => {
            chromatic_aberration(out, w, h);
            true
        }
        OverlayMode::Vhs => {
            vhs(out, w, h, t);
            true
        }
        OverlayMode::Dither => {
            dither(out, w);
            true
        }
        OverlayMode::Halftone => {
            halftone(out, w, h);
            true
        }
        OverlayMode::Grayscale => {
            grayscale(out);
            true
        }
        OverlayMode::Cyberpunk => {
            // Multi-pass stack in its own module (halation + tear + split).
            crate::cyberpunk::apply(out, w, h, t);
            true
        }
        _ => false,
    }
}

/// Radial RGB split: red is pushed outward by 1.2% of the distance from the
/// frame center, blue inward; green stays the sharp reference channel.
fn chromatic_aberration(out: &mut [u8], w: usize, h: usize) {
    let src = out.to_vec();
    let (wf, hf) = (w as f64, h as f64);
    let (cx, cy) = (wf * 0.5, hf * 0.5);
    const K: f64 = 0.012;
    out.par_chunks_mut(w * 3).enumerate().for_each(|(y, row)| {
        let dy = (y as f64 + 0.5 - cy) * K;
        for x in 0..w {
            let dx = (x as f64 + 0.5 - cx) * K;
            let fx = x as f64 + 0.5;
            let fy = y as f64 + 0.5;
            // Red outward: sample from just inside; blue inward: just outside.
            let r = sample_bilinear(&src, w, h, fx - dx, fy - dy);
            let b = sample_bilinear(&src, w, h, fx + dx, fy + dy);
            let i = x * 3;
            row[i] = to_byte(r[0]);
            row[i + 1] = src[i + 1];
            row[i + 2] = to_byte(b[2]);
        }
    });
}

/// Worn magnetic tape: deterministic per-band horizontal jitter, chroma smear
/// (luma stays sharp), and a bright tracking bar rolling top to bottom.
fn vhs(out: &mut [u8], w: usize, h: usize, t: f64) {
    let src = out.to_vec();
    let hf = h as f64;
    // Horizontal chroma-smear reach in pixels.
    const CHROMA_PX: f64 = 4.0;
    let frame = (t * 24.0).floor() as u32;
    let bar_center = (t * 0.15).fract();
    out.par_chunks_mut(w * 3).enumerate().for_each(|(y, row)| {
        let uv_y = (y as f64 + 0.5) / hf;
        let d_bar = wrap_dist(uv_y, bar_center);
        let in_bar = d_bar < 0.03;
        // Each 4-row band gets its own offset, reseeded 24 times per second.
        let seed = hash01(
            (y as u32 / 4)
                .wrapping_mul(0x9E37_79B1)
                .wrapping_add(frame.wrapping_mul(0x85EB_CA77)),
        );
        let amp = if in_bar { 14.0 } else { 3.0 } * (0.6 + 0.8 * seed as f64);
        let jitter =
            (seed as f64 - 0.5) * 2.0 * amp + (y as f64 * 0.08 + t * 9.0).sin() * amp * 0.4;
        let mut gain = 1.0 + 0.05 * (uv_y * std::f64::consts::TAU * 3.0 - t * 3.0).sin();
        if in_bar {
            gain += 0.35 * (1.0 - d_bar / 0.03);
        }
        for x in 0..w {
            let fx = x as f64 + 0.5 + jitter;
            let fy = y as f64 + 0.5;
            let sharp = sample_bilinear(&src, w, h, fx, fy);
            let left = sample_bilinear(&src, w, h, fx - CHROMA_PX, fy);
            let right = sample_bilinear(&src, w, h, fx + CHROMA_PX, fy);
            let smeared = [
                (left[0] + sharp[0] + right[0]) / 3.0,
                (left[1] + sharp[1] + right[1]) / 3.0,
                (left[2] + sharp[2] + right[2]) / 3.0,
            ];
            // Recombine sharp luma with smeared chroma: the color difference
            // is what bleeds, brightness edges stay crisp.
            let y_sharp = dot_luma(sharp);
            let y_smeared = dot_luma(smeared);
            let i = x * 3;
            for c in 0..3 {
                row[i + c] = to_byte((smeared[c] + (y_sharp - y_smeared)) * gain as f32);
            }
        }
    });
}

/// 4x4 ordered Bayer matrix (classic Macintosh values, threshold order).
const BAYER4: [[u8; 4]; 4] = [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]];

/// 1-bit ordered dithering: luma is thresholded against the Bayer matrix,
/// giving a two-tone macintosh/newspaper look with textured gradients.
fn dither(out: &mut [u8], w: usize) {
    out.par_chunks_mut(w * 3).enumerate().for_each(|(y, row)| {
        let matrix_row = &BAYER4[y % 4];
        for x in 0..w {
            let i = x * 3;
            let yv = dot_luma([row[i] as f32, row[i + 1] as f32, row[i + 2] as f32]) / 255.0;
            let threshold = (matrix_row[x % 4] as f32 + 0.5) / 16.0;
            let v = if yv > threshold { 255 } else { 0 };
            row[i] = v;
            row[i + 1] = v;
            row[i + 2] = v;
        }
    });
}

/// Halftone dot screen: one color sample per cell of a 45-degree-rotated
/// lattice, dot radius proportional to the cell luminance (linear mapping).
fn halftone(out: &mut [u8], w: usize, h: usize) {
    let (wf, hf) = (w as f64, h as f64);
    let (cx, cy) = (wf * 0.5, hf * 0.5);
    // Cell size in pixels (~90 cells on the short side), clamped so tiny
    // frames still show separate dots.
    let s = ((w.min(h) as f64) / 90.0).round().clamp(4.0, 10.0);
    // 45-degree screen angle: classic halftone, no moire against the pixel grid.
    let (cs, sn) = (
        std::f64::consts::FRAC_1_SQRT_2,
        std::f64::consts::FRAC_1_SQRT_2,
    );
    // Cell grid covering the rotated bounds of the frame.
    let half_u = (wf * cs + hf * sn) * 0.5 + s;
    let count = ((half_u / s).ceil() as i64 * 2 + 1) as usize;
    let start = -(count as i64 / 2);

    // Pass 1: sample one color per cell center (the dot keeps its cell color).
    let cells: Vec<[f32; 3]> = (0..count * count)
        .into_par_iter()
        .map(|k| {
            let i = (k % count) as i64 + start;
            let j = (k / count) as i64 + start;
            let cu = (i as f64 + 0.5) * s;
            let cv = (j as f64 + 0.5) * s;
            sample_bilinear(out, w, h, cx + cu * cs - cv * sn, cy + cu * sn + cv * cs)
        })
        .collect();

    // Pass 2: per-pixel dot coverage, colored by the owning cell.
    // The max radius reaches the cell corner (+1px anti-alias half-width), so
    // a fully bright cell tiles solid while mid-tones leave dark gaps.
    let rmax = s * std::f64::consts::FRAC_1_SQRT_2 + 0.5;
    out.par_chunks_mut(w * 3).enumerate().for_each(|(y, row)| {
        let dy = y as f64 + 0.5 - cy;
        for x in 0..w {
            let dx = x as f64 + 0.5 - cx;
            let u = dx * cs + dy * sn;
            let v = -dx * sn + dy * cs;
            let gi = (u / s).floor() as i64;
            let gj = (v / s).floor() as i64;
            let bi = (gi - start).clamp(0, count as i64 - 1) as usize;
            let bj = (gj - start).clamp(0, count as i64 - 1) as usize;
            let color = cells[bj * count + bi];
            let cu = (gi as f64 + 0.5) * s;
            let cv = (gj as f64 + 0.5) * s;
            let du = u - cu;
            let dv = v - cv;
            let d = (du * du + dv * dv).sqrt();
            // Radius is proportional to the cell luminance.
            let lum = (dot_luma(color) / 255.0).clamp(0.0, 1.0) as f64;
            let radius = lum * rmax;
            let alpha = (radius - d + 0.5).clamp(0.0, 1.0) as f32;
            let i = x * 3;
            row[i] = to_byte(color[0] * alpha);
            row[i + 1] = to_byte(color[1] * alpha);
            row[i + 2] = to_byte(color[2] * alpha);
        }
    });
}

/// Luma-weighted grayscale conversion.
fn grayscale(out: &mut [u8]) {
    out.par_chunks_mut(3).for_each(|px| {
        let v = to_byte(dot_luma([px[0] as f32, px[1] as f32, px[2] as f32]));
        px[0] = v;
        px[1] = v;
        px[2] = v;
    });
}

#[inline]
fn dot_luma(c: [f32; 3]) -> f32 {
    c[0] * LUMA[0] + c[1] * LUMA[1] + c[2] * LUMA[2]
}

#[inline]
fn to_byte(v: f32) -> u8 {
    v.round().clamp(0.0, 255.0) as u8
}

/// Shortest distance between two positions on a 0..1 wrap-around axis.
#[inline]
fn wrap_dist(a: f64, b: f64) -> f64 {
    let d = (a - b).abs();
    d.min(1.0 - d)
}

/// Bit-mixing hash in [0, 1); no global state, so parallel rows agree.
#[inline]
fn hash01(mut n: u32) -> f32 {
    n ^= n >> 16;
    n = n.wrapping_mul(0x7FEB_352D);
    n ^= n >> 15;
    n = n.wrapping_mul(0x846C_A68B);
    n ^= n >> 16;
    (n >> 8) as f32 / 16_777_216.0
}

/// Bilinear RGB sample at fractional pixel coordinates, clamped to the frame
/// (matches the GPU's linear sampler with clamp-to-edge addressing). Shared
/// with the cyberpunk stack.
#[inline]
pub(crate) fn sample_bilinear(src: &[u8], w: usize, h: usize, fx: f64, fy: f64) -> [f32; 3] {
    let fx = fx.clamp(0.0, (w - 1) as f64);
    let fy = fy.clamp(0.0, (h - 1) as f64);
    let x0 = fx.floor() as usize;
    let y0 = fy.floor() as usize;
    let tx = (fx - x0 as f64) as f32;
    let ty = (fy - y0 as f64) as f32;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let (b00, b10) = ((y0 * w + x0) * 3, (y0 * w + x1) * 3);
    let (b01, b11) = ((y1 * w + x0) * 3, (y1 * w + x1) * 3);
    let mut out = [0.0f32; 3];
    for c in 0..3 {
        let top = src[b00 + c] as f32 + (src[b10 + c] as f32 - src[b00 + c] as f32) * tx;
        let bottom = src[b01 + c] as f32 + (src[b11 + c] as f32 - src[b01 + c] as f32) * tx;
        out[c] = top + (bottom - top) * ty;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(w: usize, h: usize, rgb: [u8; 3]) -> Vec<u8> {
        rgb.iter().copied().cycle().take(w * h * 3).collect()
    }

    fn px(buf: &[u8], w: usize, x: usize, y: usize) -> [u8; 3] {
        let i = (y * w + x) * 3;
        [buf[i], buf[i + 1], buf[i + 2]]
    }

    #[test]
    fn grayscale_uses_luma_weights() {
        let mut buf = solid(4, 4, [255, 0, 0]);
        grayscale(&mut buf);
        // 0.299 * 255 = 76.2 -> 76 on every channel.
        assert!(buf.chunks(3).all(|p| p == [76, 76, 76]));
    }

    #[test]
    fn chromatic_aberration_splits_into_red_and_blue_fringes() {
        // Black frame with a white vertical stripe at x in 60..68. The split
        // grows with the distance from the center, so the stripe is placed off
        // center on a wide frame to get a few pixels of offset.
        let (w, h) = (512usize, 32usize);
        let mut buf = solid(w, h, [0, 0, 0]);
        for y in 0..h {
            for x in 60..68 {
                let i = (y * w + x) * 3;
                buf[i..i + 3].copy_from_slice(&[255, 255, 255]);
            }
        }
        chromatic_aberration(&mut buf, w, h);
        let left = px(&buf, w, 59, 16);
        let right = px(&buf, w, 68, 16);
        assert!(
            left[0] > 200 && left[2] < 50,
            "left fringe must be red-shifted, got {left:?}"
        );
        assert!(
            right[2] > 200 && right[0] < 50,
            "right fringe must be blue-shifted, got {right:?}"
        );
        // Green stays sharp: both fringe pixels are outside the stripe.
        assert_eq!(left[1], 0);
        assert_eq!(right[1], 0);
    }

    #[test]
    fn dither_quantizes_to_one_bit_and_keeps_average() {
        let mut buf = solid(4, 4, [128, 128, 128]);
        dither(&mut buf, 4);
        assert!(buf.iter().all(|&v| v == 0 || v == 255));
        let white = buf.chunks(3).filter(|p| p[0] == 255).count();
        // 128 is just above the midpoint: 8 of 16 Bayer cells round up.
        assert_eq!(white, 8, "dither average must match the source tone");
    }

    #[test]
    fn halftone_dots_scale_with_luminance() {
        // All black: no dot ink -> fully dark.
        let mut dark = solid(32, 32, [0, 0, 0]);
        halftone(&mut dark, 32, 32);
        assert!(dark.iter().all(|&v| v == 0));

        // All white: dot radius covers the whole cell -> fully lit.
        let mut bright = solid(32, 32, [255, 255, 255]);
        halftone(&mut bright, 32, 32);
        assert!(bright.iter().all(|&v| v > 200));

        // Mid gray: dot centers lit, cell corners dark.
        let mut mid = solid(32, 32, [128, 128, 128]);
        halftone(&mut mid, 32, 32);
        assert!(mid.iter().any(|&v| v > 100), "dot centers must be lit");
        assert!(mid.contains(&0), "cell gaps must stay dark");
    }

    #[test]
    fn vhs_tracking_bar_brightens_and_chroma_smears() {
        // Tracking bar starts at the top at t = 0: rows inside it are lit.
        let mut uniform = solid(32, 64, [128, 128, 128]);
        vhs(&mut uniform, 32, 64, 0.0);
        let in_bar = px(&uniform, 32, 16, 1)[0];
        let out_of_bar = px(&uniform, 32, 16, 32)[0];
        assert!(
            in_bar > out_of_bar,
            "tracking bar must brighten its rows: {in_bar} vs {out_of_bar}"
        );

        // Hard red/blue edge: chroma bleeds across it (red and blue mix on
        // both sides) while the boundary stays sharp in luma.
        let (w, h) = (64usize, 32usize);
        let mut edge = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w / 2 {
                let i = (y * w + x) * 3;
                edge[i..i + 3].copy_from_slice(&[255, 0, 0]);
            }
            for x in w / 2..w {
                let i = (y * w + x) * 3;
                edge[i..i + 3].copy_from_slice(&[0, 0, 255]);
            }
        }
        vhs(&mut edge, w, h, 0.0);
        let fringe = (20..44).any(|x| {
            let p = px(&edge, w, x, 16);
            p[0] > 80 && p[2] > 80
        });
        assert!(fringe, "chroma must bleed across the red/blue edge");
    }

    #[test]
    fn extended_modes_survive_tiny_frames() {
        for mode in [
            OverlayMode::ChromaticAberration,
            OverlayMode::Vhs,
            OverlayMode::Dither,
            OverlayMode::Halftone,
            OverlayMode::Grayscale,
            OverlayMode::Cyberpunk,
        ] {
            let mut buf = vec![120u8; 2 * 2 * 3];
            assert!(apply(&mut buf, 2, 2, mode, 0.0), "{mode:?} must be handled");
            assert_eq!(buf.len(), 12);
        }
    }

    #[test]
    fn classic_modes_fall_through() {
        let mut buf = vec![10u8; 4 * 4 * 3];
        assert!(!apply(&mut buf, 4, 4, OverlayMode::Scanlines, 0.0));
        assert!(!apply(&mut buf, 4, 4, OverlayMode::None, 0.0));
    }
}
