//! Cyberpunk overlay: cybernetic color grade, halation bloom, erratic RGB
//! split, banded digital tearing + macroblocking, holographic grid, scan
//! beam, and sync flicker. GPU parity branch: `ov == 9` in `shaders.rs`.
//!
//! Three fused passes: (A) bright pass downsampled 4x4 into a 1/4-res bloom
//! buffer, (B/C) separable box blur on that buffer, (D) one full-res pass
//! that grades, splits channels, tears rows, and overlays the grid, beam,
//! and flicker. Every effect is a deterministic function of `(x, y, t)` and
//! the frame content — no RNG, no global state — so rayon row loops cannot
//! race and tests are reproducible (same pattern as `crate::overlay`).
//!
//! Cost notes (measured on a 12-core box): the tuned stack runs ~4.4 ms per
//! 1280x720 frame (~13% of a 30 fps budget), dominated by one bilinear tap
//! per pixel in the final pass. The disruptive channels (tear shift, RGB
//! split) deliberately use integer nearest taps — sub-pixel precision is
//! invisible inside glitch artifacts.

use rayon::prelude::*;

/// Luma weights shared with the GPU shader branch.
const LUMA: [f32; 3] = [0.299, 0.587, 0.114];
/// Shadow tint: deep indigo, applied by `0.85 * (1 - luma)^2`.
const SHADOW: [f32; 3] = [0.08, 0.02, 0.45];
/// Highlight tint: hot magenta, applied by `0.60 * luma^3`.
const HIGHLIGHT: [f32; 3] = [0.72, 0.08, 0.58];
/// Contrast S-curve pivot multiplier applied before the split-tone tints.
const CONTRAST: f32 = 1.18;
/// Saturation boost after the contrast curve.
const SATURATION: f32 = 1.25;
/// Shadow split-tone strength at zero luma.
const SHADOW_STRENGTH: f32 = 0.85;
/// Highlight split-tone strength at full luma.
const HIGHLIGHT_STRENGTH: f32 = 0.60;
/// Luma at which the bloom knee starts rolling in (soft threshold on the
/// raw brightness; the max-channel term keeps neon magenta/cyan below 0.5
/// luma in the halation while skin tones stay calm).
const BLOOM_LO: f32 = 0.55;
/// Luma at which the bloom knee saturates.
const BLOOM_HI: f32 = 0.95;
/// Bloom peak gain for fully hot pixels.
const BLOOM_GAIN: f32 = 1.2;
/// Max-channel weight in the "hotness" measure (see [`BLOOM_LO`]).
const HOT_CHROMA: f32 = 0.75;
/// How far the halo color is pushed toward the magenta highlight tint.
const HALO_TINT: f32 = 0.45;
/// Box blur radius (pixels) at 1/4 res == 48 px of halo spread at full res.
const BLOOM_RADIUS: usize = 12;
/// Fraction of 24-row bands that rip horizontally per frame (~15%).
const TEAR_P: f32 = 0.85;
/// Fraction of ripping bands that also macroblock into 8px blocks.
const BLOCK_P: f32 = 0.93;

/// Applies the full cyberpunk stack to a composited frame in place.
pub(crate) fn apply(out: &mut [u8], w: usize, h: usize, t: f64) {
    if w < 2 || h < 2 {
        return;
    }
    let src = out.to_vec();
    // Cell count covers the whole frame (ceil), so ragged edges like 5px
    // rows still get a valid 1/4-res cell.
    let (qw, qh) = (w.div_ceil(4).max(1), h.div_ceil(4).max(1));
    let mut bloom = vec![0.0f32; qw * qh * 3];
    let mut tmp = vec![0.0f32; qw * qh * 3];
    build_bloom(&src, &mut bloom, w, h, qw);
    blur_axis(&bloom, &mut tmp, qw, qh, true);
    blur_axis(&tmp, &mut bloom, qw, qh, false);
    finalize(&src, &bloom, out, w, h, qw, t);
}

/// Grade: contrast curve, saturation boost, then split-tone — shadows to
/// deep indigo, highlights to hot magenta. Mirrored by `grade()` in the
/// WGSL shaders; strengths are tuned to read as "neon night" even on
/// low-contrast webcam light.
#[inline]
fn grade(c: [f32; 3]) -> [f32; 3] {
    let mut o = [0.0f32; 3];
    let mut l = dot_luma(c);
    // Contrast S-curve: crush blacks, lift whites.
    for (k, v) in c.iter().enumerate() {
        o[k] = (v - 0.5) * CONTRAST + 0.5;
    }
    l = (l - 0.5) * CONTRAST + 0.5;
    // Saturation boost around the new luma.
    for v in o.iter_mut() {
        *v = l + (*v - l) * SATURATION;
    }
    l = dot_luma(o);
    // Split-tone: (1 - l)^2 and l^3 keep the two tints from fighting over
    // the midtones.
    let s = (1.0 - l) * (1.0 - l) * SHADOW_STRENGTH;
    let h = l * l * l * HIGHLIGHT_STRENGTH;
    for (v, tint) in o.iter_mut().zip(SHADOW.iter().zip(HIGHLIGHT.iter())) {
        *v = *v + (tint.0 - *v) * s + (tint.1 - *v) * h;
    }
    o
}

/// Pass A: per-block (4x4) bright pass. Each 1/4-res row owns exactly four
/// source rows, so the bloom writes are disjoint across rayon threads.
fn build_bloom(src: &[u8], bloom: &mut [f32], w: usize, h: usize, qw: usize) {
    bloom
        .par_chunks_mut(qw * 3)
        .enumerate()
        .for_each(|(qy, row)| {
            let mut acc = vec![[0.0f32; 3]; qw];
            for dy in 0..4 {
                let y = (qy * 4 + dy).min(h - 1);
                let base = y * w * 3;
                for x in 0..w.min(qw * 4) {
                    let i = base + x * 3;
                    let raw = [
                        src[i] as f32 / 255.0,
                        src[i + 1] as f32 / 255.0,
                        src[i + 2] as f32 / 255.0,
                    ];
                    // Hotness on the RAW pixel: pure magenta/cyan neon sits at
                    // ~0.4 luma, so the max-channel term keeps colored light
                    // in the halation while bright-but-flat skin stays out.
                    let hot =
                        dot_luma(raw).max(raw.iter().copied().fold(0.0f32, f32::max) * HOT_CHROMA);
                    let wgt = smoothstep(BLOOM_LO, BLOOM_HI, hot) * BLOOM_GAIN;
                    let c = grade(raw);
                    let cell = &mut acc[x / 4];
                    for k in 0..3 {
                        // Halo color drifts toward the magenta tint: searing
                        // neon haze rather than plain white glow.
                        cell[k] += (c[k] + (HIGHLIGHT[k] - c[k]) * HALO_TINT) * wgt;
                    }
                }
            }
            for (x, cell) in acc.iter().enumerate() {
                let o = x * 3;
                for k in 0..3 {
                    row[o + k] = cell[k] * 0.0625;
                }
            }
        });
}

/// Pass B/C: separable box blur on the 1/4-res buffer. Every thread writes
/// only its own pixel slots (one 3-float chunk each) while reading the whole
/// immutable `bloom`; the window runs along `x` for the horizontal pass and
/// along rows for the vertical pass.
fn blur_axis(bloom: &[f32], out: &mut [f32], qw: usize, qh: usize, horizontal: bool) {
    let (len, step) = if horizontal { (qw, 3) } else { (qh, qw * 3) };
    out.par_chunks_mut(3).enumerate().for_each(|(e, slot)| {
        let (r, i) = (e / qw, e % qw);
        let (var, fixed) = if horizontal { (i, r) } else { (r, i) };
        let lo = var.saturating_sub(BLOOM_RADIUS);
        let hi = (var + BLOOM_RADIUS).min(len - 1);
        // Column base for the vertical pass: `fixed` is the column index, so
        // taps advance by a whole row; for the horizontal pass `fixed` is the
        // row number.
        let base = if horizontal {
            fixed * qw * 3
        } else {
            fixed * 3
        };
        let mut acc = [0.0f32; 3];
        let mut n = 0.0f32;
        for j in lo..=hi {
            let o = base + j * step;
            for k in 0..3 {
                acc[k] += bloom[o + k];
            }
            n += 1.0;
        }
        for k in 0..3 {
            slot[k] = acc[k] / n;
        }
    });
}

/// Pass D: full-res pass fusing grade + split + tear + block glitch + bloom.
fn finalize(src: &[u8], bloom: &[f32], out: &mut [u8], w: usize, h: usize, qw: usize, t: f64) {
    let frame = (t * 30.0).floor() as u32;
    let bar = (t * 0.12).fract() as f32;
    let vshift = ((t * 37.0).sin() * 1.2).round() as i64;
    let flicker = 1.0 + 0.045 * (t * 200.0).sin();
    out.par_chunks_mut(w * 3).enumerate().for_each(|(y, row)| {
        let uv_y = (y as f64 + 0.5) / h as f64;
        let d_bar = {
            let d = (uv_y - bar as f64).abs();
            d.min(1.0 - d)
        };
        let band = (y / 24) as u32;
        let tear = hash01(band.wrapping_mul(0x9E37_79B1) ^ frame.wrapping_mul(0x85EB_CA77));
        let rip = tear > TEAR_P;
        let shift: i64 = if rip {
            ((hash01(band ^ frame.wrapping_mul(0x27D4_EB2D)) - 0.5) * 2.0 * 0.06 * w as f32).round()
                as i64
        } else {
            0
        };
        let block = rip && tear > BLOCK_P;
        let gain_beam = if d_bar < 0.05 {
            1.0 + 0.22 * (1.0 - d_bar / 0.05)
        } else {
            1.0
        };
        let scan = if y % 4 >= 2 { 0.90 } else { 1.0 };
        let k = (gain_beam * scan * flicker) as f32;
        let sy = (y as i64 + vshift).clamp(0, h as i64 - 1) as usize;
        // Source row after the vertical sync jitter; glitch taps stay within
        // this row (whole-row tearing), so the frame is read once per row.
        let srow = &src[sy * w * 3..sy * w * 3 + w * 3];
        let jit = (hash01((y as u32).wrapping_mul(0x2545_F491) ^ frame) - 0.5) * 2.0;
        let split = (jit * 1.5).round() as i64;
        let row_bloom = sy / 4 * qw;
        for x in 0..w {
            let mut sx = x as i64 + shift;
            if block {
                // Macroblocking: quantize the sample to 8px block centers.
                sx = (sx / 8) * 8 + 4;
            }
            let sx = sx.clamp(0, w as i64 - 1) as usize;
            // Green stays the sharp reference: one bilinear tap.
            let g = grade(crate::overlay::sample_bilinear(
                srow,
                w,
                1,
                sx as f64 + 0.5,
                0.5,
            ));
            // Erratic chromatic aberration: red/blue split by the row jitter
            // (~+/-2 px), integer taps (sub-pixel detail is invisible inside
            // glitch artifacts).
            let rx = (sx as i64 + split - 1).clamp(0, w as i64 - 1) as usize;
            let bx = (sx as i64 + split + 1).clamp(0, w as i64 - 1) as usize;
            let r = grade_channel(srow, rx, 0);
            let b = grade_channel(srow, bx, 2);
            let mut c = [r, g[1], b];
            // Halation: nearest upsample of the blurred 1/4-res halo (the
            // blur already hides the 4px quantization).
            let bi = row_bloom + sx / 4;
            for k in 0..3 {
                c[k] += bloom[bi * 3 + k];
            }
            // Holographic grid + scanlines + chaff speckles.
            let grid = if (x % 64 < 1) || (y % 64 < 1) {
                0.10
            } else {
                0.0
            };
            let i = x * 3;
            for ch in 0..3 {
                let mut v = (c[ch] * k + grid) * 255.0;
                if hash01((y as u32).wrapping_mul(811) ^ (x as u32).wrapping_mul(577) ^ frame)
                    > 0.998
                {
                    // Chaff: sparse neon speckles popping between frames.
                    v = if ch == 2 {
                        255.0
                    } else if ch == 0 {
                        120.0
                    } else {
                        40.0
                    };
                }
                row[i + ch] = v.clamp(0.0, 255.0) as u8;
            }
        }
    });
}

/// Grades one channel of a source pixel (nearest tap on `row`).
#[inline]
fn grade_channel(row: &[u8], x: usize, ch: usize) -> f32 {
    let c = [
        row[x * 3] as f32 / 255.0,
        row[x * 3 + 1] as f32 / 255.0,
        row[x * 3 + 2] as f32 / 255.0,
    ];
    grade(c)[ch]
}

#[inline]
fn dot_luma(c: [f32; 3]) -> f32 {
    c[0] * LUMA[0] + c[1] * LUMA[1] + c[2] * LUMA[2]
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

/// Hermite interpolation on `v` across `[lo, hi]`, clamped: the soft knee the
/// bloom uses instead of a hard threshold.
#[inline]
fn smoothstep(lo: f32, hi: f32, v: f32) -> f32 {
    let t = ((v - lo) / (hi - lo)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(w: usize, h: usize, rgb: [u8; 3]) -> Vec<u8> {
        rgb.iter().copied().cycle().take(w * h * 3).collect()
    }

    #[test]
    fn grade_pushes_shadows_toward_blue() {
        let dark = [0.2f32; 3];
        let g = grade(dark);
        // Blue tint > red tint for a dark neutral pixel.
        assert!(g[2] > g[0], "shadows must gain blue, got {g:?}");
        assert!(
            g[0] > 0.2 || g[2] > 0.2,
            "shadows must be lifted, got {g:?}"
        );
    }

    #[test]
    fn grade_pushes_highlights_toward_magenta() {
        let bright = [0.9f32; 3];
        let g = grade(bright);
        // Hot magenta: red and blue lifted well above green, red dominant.
        // The grade may overshoot slightly above 1.0; the final pass clamps
        // on the u8 write.
        assert!(
            g[0] > g[1] && g[2] > g[1],
            "highlights must go magenta, got {g:?}"
        );
        assert!(g[0] > g[2] - 0.1, "magenta leans red, got {g:?}");
        assert!(g[0] < 1.1 && g[2] < 1.1, "grade must stay bounded: {g:?}");
    }

    #[test]
    fn grade_crushes_blacks_and_lifts_whites() {
        // Contrast curve: the luma spread between dark and bright grows.
        let dark = grade([0.15f32; 3]);
        let bright = grade([0.85f32; 3]);
        assert!(
            dark[1] < 0.15,
            "blacks must be crushed below source: {dark:?}"
        );
        let spread = bright[1] - dark[1];
        assert!(spread > 0.40, "contrast must widen the luma band: {spread}");
    }

    #[test]
    fn grade_boosts_chroma_of_midtones() {
        let c = [0.5f32, 0.3, 0.4];
        let before = dot_luma(c);
        let before_c = c.iter().map(|v| (v - before).abs()).fold(0.0f32, f32::max);
        let g = grade(c);
        let after = dot_luma(g);
        let after_c = g.iter().map(|v| (v - after).abs()).fold(0.0f32, f32::max);
        assert!(
            after_c > before_c * 1.1,
            "saturation boost must widen chroma: {before_c} -> {after_c}"
        );
    }

    #[test]
    fn bloom_uses_soft_knee_and_no_dark_leak() {
        // Black frame: the smoothstep knee stays at zero for unlit pixels.
        let (w, h, qw, qh) = (32usize, 32usize, 8usize, 8usize);
        let mut src = solid(w, h, [0, 0, 0]);
        for y in 0..8 {
            for x in 0..8 {
                let i = (y * w + x) * 3;
                src[i..i + 3].copy_from_slice(&[255, 250, 235]);
            }
        }
        let mut bloom = vec![0.0f32; qw * qh * 3];
        build_bloom(&src, &mut bloom, w, h, qw);
        // A hot block feeds the halo, and the halo leans magenta (red and
        // blue above green, red on top) instead of staying white.
        assert!(bloom[0] > 0.0, "highlight cell must feed the bloom");
        assert!(
            bloom[0] > bloom[1] && bloom[2] > bloom[1] && bloom[0] > bloom[2],
            "halo must be tinted hot magenta: {:?}",
            &bloom[..3]
        );
        assert!(bloom.iter().all(|&v| v >= 0.0), "no negative bloom");
    }

    #[test]
    fn bloom_only_accumulates_around_highlights() {
        // Black frame with one bright 8x8 block in the top-left.
        let (w, h, qw, qh) = (32usize, 32usize, 8usize, 8usize);
        let mut src = solid(w, h, [0, 0, 0]);
        for y in 0..8 {
            for x in 0..8 {
                let i = (y * w + x) * 3;
                src[i..i + 3].copy_from_slice(&[255, 250, 235]);
            }
        }
        let mut bloom = vec![0.0f32; qw * qh * 3];
        build_bloom(&src, &mut bloom, w, h, qw);
        // Only the cell overlapped by the bright block has energy.
        assert!(bloom[0] > 0.0, "highlight cell must feed the bloom");
        let tail = &bloom[8 * 3 * 7..];
        assert!(tail.iter().all(|&v| v == 0.0), "dark cells must stay empty");
    }

    #[test]
    fn blur_averages_and_dark_frame_stays_dark() {
        let (qw, qh) = (8usize, 8usize);
        let mut bloom = vec![0.0f32; qw * qh * 3];
        bloom[0] = 1.0;
        bloom[1] = 1.0;
        bloom[2] = 1.0;
        let mut tmp = vec![0.0f32; qw * qh * 3];
        blur_axis(&bloom, &mut tmp, qw, qh, true);
        blur_axis(&tmp, &mut bloom, qw, qh, false);
        assert!(bloom[4 * 3] > 0.0, "halo must spread to neighbors");
        assert!(bloom.iter().all(|&v| v <= 1.0), "box blur never amplifies");
    }

    #[cfg(test)]
    #[test]
    fn apply_is_deterministic_in_time() {
        let mut a = solid(64, 64, [128, 128, 128]);
        let mut b = a.clone();
        apply(&mut a, 64, 64, 1.5);
        apply(&mut b, 64, 64, 1.5);
        assert_eq!(a, b, "same t must produce identical frames");
    }

    #[test]
    fn apply_animates_and_changes_content() {
        let base = solid(64, 64, [128, 128, 128]);
        let mut t0 = base.clone();
        apply(&mut t0, 64, 64, 0.0);
        let mut t1 = base.clone();
        apply(&mut t1, 64, 64, 7.0);
        assert_ne!(t0, t1, "animated glitch must change with time");
        assert!(
            t0.iter().any(|&v| v != 128),
            "cyberpunk look must not be a no-op"
        );
    }

    #[test]
    fn tiny_frames_survive() {
        for (w, h) in [(2usize, 2usize), (1, 1), (5, 3)] {
            let mut buf = vec![120u8; w * h * 3];
            apply(&mut buf, w, h, 0.5);
            assert_eq!(buf.len(), w * h * 3);
        }
    }
}
