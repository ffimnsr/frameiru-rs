//! Fast pixel conversions for sink outputs.
//!
//! `rgb8_to_yuyv422` is the inverse of `frameiru_capture::decoders::yuyv_to_rgb8`
//! (same ITU-R BT.601 limited-range matrix), so roundtrips lose only
//! sub-quantum rounding error.
//!
//! On x86_64 the hot path uses SSSE3 with a scalar fallback; both paths are
//! bit-exact against each other (verified by tests).

use frameiru_core::error::FrameiruError;

/// Converts a tightly packed RGB24 frame into YUYV 4:2:2.
///
/// `input` must hold `width * height * 3` bytes; `output` must hold
/// `ceil(width / 2) * 4 * height` bytes (for even widths, `width * height *
/// 2`). Chroma (U/V) is averaged over each horizontal pixel pair; an
/// odd-width frame pads the trailing pair by duplicating the last pixel.
/// Uses the BT.601 limited-range forward matrix (fixed point, /256).
///
/// Auto-selects the SSSE3 SIMD path on x86_64 when available.
pub fn rgb8_to_yuyv422(
    input: &[u8],
    width: u32,
    height: u32,
    output: &mut [u8],
) -> Result<(), FrameiruError> {
    let w = width as usize;
    let h = height as usize;
    if width == 0 || height == 0 {
        return Err(FrameiruError::InvalidArgument(
            "RGB dimensions must be non-zero".into(),
        ));
    }
    let need_in = w * h * 3;
    let need_out = (w.div_ceil(2)) * 4 * h;
    if input.len() < need_in {
        return Err(FrameiruError::InsufficientCapacity {
            need: need_in,
            have: input.len(),
        });
    }
    if output.len() < need_out {
        return Err(FrameiruError::InsufficientCapacity {
            need: need_out,
            have: output.len(),
        });
    }

    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("ssse3") {
            // SAFETY: input/output slices are length-checked above and the
            // SSSE3 feature was just verified.
            unsafe { yuyv_ssse3(input, w, h, output) }
        } else {
            yuyv_scalar(input, w, h, output);
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    yuyv_scalar(input, w, h, output);

    Ok(())
}

/// Scalar reference implementation (unchecked sizes).
fn yuyv_scalar(input: &[u8], w: usize, h: usize, output: &mut [u8]) {
    for row in 0..h {
        let in_row = &input[row * w * 3..(row + 1) * w * 3];
        let out_row = &mut output[row * (w.div_ceil(2)) * 4..(row + 1) * (w.div_ceil(2)) * 4];
        for x in (0..w).step_by(2) {
            // Odd trailing pixel pairs with itself.
            let x1 = (x + 1).min(w - 1);
            let (r0, g0, b0) = (in_row[x * 3], in_row[x * 3 + 1], in_row[x * 3 + 2]);
            let (r1, g1, b1) = (in_row[x1 * 3], in_row[x1 * 3 + 1], in_row[x1 * 3 + 2]);
            let (u, v) = rgb_to_uv(
                ((u32::from(r0) + u32::from(r1)) / 2) as u8,
                ((u32::from(g0) + u32::from(g1)) / 2) as u8,
                ((u32::from(b0) + u32::from(b1)) / 2) as u8,
            );
            let i = (x / 2) * 4;
            out_row[i] = rgb_to_y(r0, g0, b0);
            out_row[i + 1] = u;
            out_row[i + 2] = rgb_to_y(r1, g1, b1);
            out_row[i + 3] = v;
        }
    }
}

/// BT.601 limited range luma, fixed-point coefficients scaled by /256.
fn rgb_to_y(r: u8, g: u8, b: u8) -> u8 {
    let (r, g, b) = (r as i32, g as i32, b as i32);
    clamp_u8(((66 * r + 129 * g + 25 * b + 128) >> 8) + 16)
}

/// BT.601 limited range chroma, fixed-point coefficients scaled by /256.
fn rgb_to_uv(r: u8, g: u8, b: u8) -> (u8, u8) {
    let (r, g, b) = (r as i32, g as i32, b as i32);
    let u = ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
    let v = ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
    (clamp_u8(u), clamp_u8(v))
}

fn clamp_u8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// SSSE3-accelerated conversion (unchecked sizes). Bit-exact with
/// [`yuyv_scalar`]: chroma is pair-averaged with floor division and the
/// BT.601 matrix is applied in centered (r-128) form so all intermediate
/// sums fit in i16/i32 exactly like the scalar i32 math.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
unsafe fn yuyv_ssse3(input: &[u8], w: usize, h: usize, output: &mut [u8]) {
    use std::arch::x86_64::*;

    // Deinterleave masks: extract the r/g/b byte of each pixel from the
    // three overlapping 16-byte loads covering a 48-byte (16 pixel) group.
    const M_R0: [i8; 16] = [
        0, 3, 6, 9, 12, 15, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
    ];
    const M_R1: [i8; 16] = [
        -128, -128, -128, -128, -128, -128, 2, 5, 8, 11, 14, -128, -128, -128, -128, -128,
    ];
    const M_R2: [i8; 16] = [
        -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, 1, 4, 7, 10, 13,
    ];
    const M_G0: [i8; 16] = [
        1, 4, 7, 10, 13, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
    ];
    const M_G1: [i8; 16] = [
        -128, -128, -128, -128, -128, 0, 3, 6, 9, 12, 15, -128, -128, -128, -128, -128,
    ];
    const M_G2: [i8; 16] = [
        -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, 2, 5, 8, 11, 14,
    ];
    const M_B0: [i8; 16] = [
        2, 5, 8, 11, 14, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
    ];
    const M_B1: [i8; 16] = [
        -128, -128, -128, -128, -128, 1, 4, 7, 10, 13, -128, -128, -128, -128, -128, -128,
    ];
    const M_B2: [i8; 16] = [
        -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, 0, 3, 6, 9, 12, 15,
    ];
    // Split 16 packed channel bytes into even-indexed (low 8) and
    // odd-indexed (high 8) bytes for pair averaging.
    const M_EVEN_ODD: [i8; 16] = [0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15];
    const COEF_Y_RG: [i16; 8] = [66, 129, 66, 129, 66, 129, 66, 129];
    const COEF_Y_B: [i16; 8] = [25, 0, 25, 0, 25, 0, 25, 0];
    const COEF_U_RG: [i16; 8] = [-38, -74, -38, -74, -38, -74, -38, -74];
    const COEF_U_B: [i16; 8] = [112, 0, 112, 0, 112, 0, 112, 0];
    const COEF_V_RG: [i16; 8] = [112, -94, 112, -94, 112, -94, 112, -94];
    const COEF_V_B: [i16; 8] = [-18, 0, -18, 0, -18, 0, -18, 0];

    let load = |p: *const __m128i| _mm_loadu_si128(p);
    let shuffle = |v: __m128i, m: __m128i| _mm_shuffle_epi8(v, m);
    let z = _mm_setzero_si128();
    let center = _mm_set1_epi16(128);
    let y_bias = _mm_set1_epi32(28288); // 66*128 + 129*128 + 25*128 + 128
    let u_bias = _mm_set1_epi32(32896); // (X + 128) >> 8 + 128 = (X + 128 + 32768) >> 8
    let m_r0 = load(M_R0.as_ptr() as *const __m128i);
    let m_r1 = load(M_R1.as_ptr() as *const __m128i);
    let m_r2 = load(M_R2.as_ptr() as *const __m128i);
    let m_g0 = load(M_G0.as_ptr() as *const __m128i);
    let m_g1 = load(M_G1.as_ptr() as *const __m128i);
    let m_g2 = load(M_G2.as_ptr() as *const __m128i);
    let m_b0 = load(M_B0.as_ptr() as *const __m128i);
    let m_b1 = load(M_B1.as_ptr() as *const __m128i);
    let m_b2 = load(M_B2.as_ptr() as *const __m128i);
    let coeff_y_rg = load(COEF_Y_RG.as_ptr() as *const __m128i);
    let coeff_y_b = load(COEF_Y_B.as_ptr() as *const __m128i);
    let coeff_u_rg = load(COEF_U_RG.as_ptr() as *const __m128i);
    let coeff_u_b = load(COEF_U_B.as_ptr() as *const __m128i);
    let coeff_v_rg = load(COEF_V_RG.as_ptr() as *const __m128i);
    let coeff_v_b = load(COEF_V_B.as_ptr() as *const __m128i);
    let even_odd = load(M_EVEN_ODD.as_ptr() as *const __m128i);

    for row in 0..h {
        let in_row = &input[row * w * 3..(row + 1) * w * 3];
        let out_row = &mut output[row * (w.div_ceil(2)) * 4..(row + 1) * (w.div_ceil(2)) * 4];

        let mut x = 0;
        while x + 16 <= w {
            let p = in_row[x * 3..].as_ptr();
            let l0 = load(p as *const __m128i);
            let l1 = load(p.add(16) as *const __m128i);
            let l2 = load(p.add(32) as *const __m128i);

            // 16 channel bytes per component.
            let r = _mm_or_si128(
                _mm_or_si128(shuffle(l0, m_r0), shuffle(l1, m_r1)),
                shuffle(l2, m_r2),
            );
            let g = _mm_or_si128(
                _mm_or_si128(shuffle(l0, m_g0), shuffle(l1, m_g1)),
                shuffle(l2, m_g2),
            );
            let b = _mm_or_si128(
                _mm_or_si128(shuffle(l0, m_b0), shuffle(l1, m_b1)),
                shuffle(l2, m_b2),
            );

            // Luma per pixel: Y = (66r + 129g + 25b + 128) >> 8 + 16, using
            // centered channels so i16 math cannot overflow. Four batches of
            // four pixels (0..3, 4..7, 8..11, 12..15) keep lane order exact.
            let r16_lo = _mm_sub_epi16(_mm_unpacklo_epi8(r, z), center);
            let r16_hi = _mm_sub_epi16(_mm_unpackhi_epi8(r, z), center);
            let g16_lo = _mm_sub_epi16(_mm_unpacklo_epi8(g, z), center);
            let g16_hi = _mm_sub_epi16(_mm_unpackhi_epi8(g, z), center);
            let b16_lo = _mm_sub_epi16(_mm_unpacklo_epi8(b, z), center);
            let b16_hi = _mm_sub_epi16(_mm_unpackhi_epi8(b, z), center);
            let y_batch = |rgiv: __m128i, biv: __m128i| {
                _mm_add_epi32(
                    _mm_madd_epi16(rgiv, coeff_y_rg),
                    _mm_madd_epi16(biv, coeff_y_b),
                )
            };
            let y0 = y_batch(
                _mm_unpacklo_epi16(r16_lo, g16_lo),
                _mm_unpacklo_epi16(b16_lo, z),
            );
            let y1 = y_batch(
                _mm_unpackhi_epi16(r16_lo, g16_lo),
                _mm_unpackhi_epi16(b16_lo, z),
            );
            let y2 = y_batch(
                _mm_unpacklo_epi16(r16_hi, g16_hi),
                _mm_unpacklo_epi16(b16_hi, z),
            );
            let y3 = y_batch(
                _mm_unpackhi_epi16(r16_hi, g16_hi),
                _mm_unpackhi_epi16(b16_hi, z),
            );
            let y01 = _mm_packus_epi32(
                _mm_srai_epi32(_mm_add_epi32(y0, y_bias), 8),
                _mm_srai_epi32(_mm_add_epi32(y1, y_bias), 8),
            );
            let y23 = _mm_packus_epi32(
                _mm_srai_epi32(_mm_add_epi32(y2, y_bias), 8),
                _mm_srai_epi32(_mm_add_epi32(y3, y_bias), 8),
            );
            let y = _mm_add_epi8(_mm_packus_epi16(y01, y23), _mm_set1_epi8(16));

            // Pair-average channels (floor division, matching scalar).
            let avg = |c: __m128i| {
                let split = shuffle(c, even_odd);
                let even = _mm_unpacklo_epi8(split, z);
                let odd = _mm_unpackhi_epi8(split, z);
                _mm_packus_epi16(_mm_srai_epi16(_mm_add_epi16(even, odd), 1), z)
            };
            let ravg = avg(r);
            let gavg = avg(g);
            let bavg = avg(b);

            // Chroma per pair from the averaged channels: coeffs sum to zero,
            // so no centering bias is needed beyond the +128 output offset.
            let uv = |rg: __m128i,
                      gch: __m128i,
                      bch: __m128i,
                      coeff_rg: __m128i,
                      coeff_b: __m128i|
             -> __m128i {
                let rl = _mm_sub_epi16(_mm_unpacklo_epi8(rg, z), center);
                let rh = _mm_sub_epi16(_mm_unpackhi_epi8(rg, z), center);
                let gl = _mm_sub_epi16(_mm_unpacklo_epi8(gch, z), center);
                let gh = _mm_sub_epi16(_mm_unpackhi_epi8(gch, z), center);
                let bl = _mm_sub_epi16(_mm_unpacklo_epi8(bch, z), center);
                let bh = _mm_sub_epi16(_mm_unpackhi_epi8(bch, z), center);
                let batch = |rgiv: __m128i, biv: __m128i| {
                    _mm_add_epi32(_mm_madd_epi16(rgiv, coeff_rg), _mm_madd_epi16(biv, coeff_b))
                };
                let shift = |c: __m128i| _mm_srai_epi32(_mm_add_epi32(c, u_bias), 8);
                let c01 = _mm_packus_epi32(
                    shift(batch(_mm_unpacklo_epi16(rl, gl), _mm_unpacklo_epi16(bl, z))),
                    shift(batch(_mm_unpackhi_epi16(rl, gl), _mm_unpackhi_epi16(bl, z))),
                );
                let c23 = _mm_packus_epi32(
                    shift(batch(_mm_unpacklo_epi16(rh, gh), _mm_unpacklo_epi16(bh, z))),
                    shift(batch(_mm_unpackhi_epi16(rh, gh), _mm_unpackhi_epi16(bh, z))),
                );
                _mm_packus_epi16(c01, c23)
            };
            let u = uv(ravg, gavg, bavg, coeff_u_rg, coeff_u_b);
            let v = uv(ravg, gavg, bavg, coeff_v_rg, coeff_v_b);

            // Interleave [Y_even | U] and [Y_odd | V] into [Y0 U Y1 V] pairs.
            // After the even/odd shuffle the low 8 bytes hold even luma, the
            // high 8 odd luma; shift the high half down before interleaving.
            let y_split = shuffle(y, even_odd);
            let y_odd = _mm_srli_si128(y_split, 8);
            let t1 = _mm_unpacklo_epi8(y_split, u);
            let t2 = _mm_unpacklo_epi8(y_odd, v);
            _mm_storeu_si128(
                out_row[x * 2..].as_mut_ptr() as *mut __m128i,
                _mm_unpacklo_epi16(t1, t2),
            );
            _mm_storeu_si128(
                out_row[x * 2 + 16..].as_mut_ptr() as *mut __m128i,
                _mm_unpackhi_epi16(t1, t2),
            );

            x += 16;
        }

        if x < w {
            // Remaining pixels (includes the odd-width self-pair).
            yuyv_scalar(&in_row[x * 3..], w - x, 1, &mut out_row[x * 2..]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use frameiru_capture::decoders::yuyv_to_rgb8;

    /// Deterministic pseudo-random input (xorshift) sized for `w x h`.
    fn random_rgb(w: usize, h: usize, seed: u64) -> Vec<u8> {
        let mut state = seed | 1;
        (0..w * h * 3)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state & 0xff) as u8
            })
            .collect()
    }

    #[test]
    fn gray_roundtrips_through_yuyv() {
        let w = 8u32;
        let h = 4u32;
        // Mid gray (128, 128, 128) is exactly representable in limited range.
        let rgb = vec![128u8; (w * h * 3) as usize];
        let mut yuyv = vec![0u8; (w * h * 2) as usize];
        rgb8_to_yuyv422(&rgb, w, h, &mut yuyv).unwrap();
        let mut back = vec![0u8; (w * h * 3) as usize];
        yuyv_to_rgb8(&yuyv, w, h, &mut back).unwrap();
        // Lossy only in rounding: within 2 quantization steps of 128.
        assert!(
            back.iter().all(|&v| (v as i32 - 128).abs() <= 2),
            "roundtrip drift too large: {back:?}"
        );
    }

    #[test]
    fn paired_pixels_share_averaged_chroma() {
        // Red + blue pair: chroma is the average (128, 0, 128).
        let rgb = [255, 0, 0, 0, 0, 255];
        let mut yuyv = vec![0u8; 4];
        rgb8_to_yuyv422(&rgb, 2, 1, &mut yuyv).unwrap();
        assert_ne!(yuyv[0], yuyv[2], "Y must differ between red and blue");
        assert_eq!(yuyv[1], 165, "U of (128,0,128) under BT.601");
        assert_eq!(yuyv[3], 175, "V of (128,0,128) under BT.601");
    }

    #[test]
    fn odd_width_pads_trailing_pair_with_itself() {
        // 3 pixels: pairs are (red, green) and (blue, blue).
        let rgb = [255, 0, 0, 0, 255, 0, 0, 0, 255];
        let mut yuyv = vec![0u8; 8];
        rgb8_to_yuyv422(&rgb, 3, 1, &mut yuyv).unwrap();
        assert_eq!(yuyv.len(), 8, "odd width emits a padded pair");
        assert_eq!(yuyv[4], yuyv[6], "trailing pixel duplicates its Y");
        assert_eq!(yuyv[5], 240, "U of pure blue under BT.601");
        assert_eq!(yuyv[7], 110, "V of pure blue under BT.601");
    }

    #[test]
    fn checks_buffer_sizes() {
        let rgb = vec![0u8; 12];
        assert!(matches!(
            rgb8_to_yuyv422(&rgb, 2, 2, &mut [0u8; 4]),
            Err(FrameiruError::InsufficientCapacity { .. })
        ));
        assert!(matches!(
            rgb8_to_yuyv422(&[0u8; 5], 2, 2, &mut [0u8; 8]),
            Err(FrameiruError::InsufficientCapacity { .. })
        ));
        // Odd width needs ceil(3/2)*4 = 8 bytes, not 6.
        assert!(matches!(
            rgb8_to_yuyv422(&[0u8; 9], 3, 1, &mut [0u8; 6]),
            Err(FrameiruError::InsufficientCapacity { .. })
        ));
    }

    #[test]
    fn rejects_zero_dimensions() {
        assert!(rgb8_to_yuyv422(&[], 0, 2, &mut []).is_err());
        assert!(rgb8_to_yuyv422(&[], 2, 0, &mut []).is_err());
    }

    /// The SIMD path must be bit-exact with the scalar reference on every
    /// shape, including odd widths and rows not aligned to 16 pixels.
    /// The SIMD path must be bit-exact with the scalar reference on every
    /// shape, including odd widths and rows not aligned to 16 pixels.
    #[test]
    fn simd_matches_scalar_exactly() {
        for &(w, h) in &[
            (2, 1),
            (3, 1),
            (15, 2),
            (16, 1),
            (17, 3),
            (31, 2),
            (32, 2),
            (33, 4),
            (64, 48),
            (100, 7),
        ] {
            let rgb = random_rgb(w, h, (w * 31 + h) as u64);
            let mut fast = vec![0u8; (w.div_ceil(2)) * 4 * h];
            let mut slow = vec![0u8; (w.div_ceil(2)) * 4 * h];
            rgb8_to_yuyv422(&rgb, w as u32, h as u32, &mut fast).unwrap();
            yuyv_scalar(&rgb, w, h, &mut slow);
            assert_eq!(fast, slow, "SIMD/scalar mismatch at {w}x{h}");
        }
    }
}
