//! Mask polish: dilation + feather + soft threshold applied after
//! postprocessing so the composite never shows sharp source pixels where
//! the model hesitated, without hardening the subject edge.
//!
//! MediaPipe-class masks at 256x144 mislabel a few percent of edge regions
//! (hair, glasses, hands); those pixels keep `alpha` near 1 and the
//! composite shows the *sharp* source instead of the blur. Dilating the
//! foreground (3x3 gray max) covers the leak, a follow-up box feather
//! restores the soft transition band (plain dilation alone max-filters the
//! matte edge to 1.0, which reads as an unblurred ring around the subject),
//! and an optional soft threshold collapses ghosting mid-alphas toward 0/1.

use frameiru_core::buffer::Mask;
use frameiru_core::error::FrameiruError;

/// Applies foreground dilation (`dilate` 3x3-max iterations), a matching
/// box feather (`radius = dilate`), and an optional soft threshold
/// (`contrast` center, band 0.15) to `mask` in place.
/// `dilate == 0 && contrast == 0.0` is a no-op.
pub fn polish_mask(mask: &mut Mask, dilate: u32, contrast: f32) -> Result<(), FrameiruError> {
    if mask.data.is_empty() || !mask.resolution.is_valid() {
        return Err(FrameiruError::InvalidArgument(
            "polish requires a valid mask".into(),
        ));
    }
    if dilate > 0 {
        let w = mask.resolution.width as usize;
        let h = mask.resolution.height as usize;
        let mut scratch = vec![0f32; mask.data.len()];
        for _ in 0..dilate {
            scratch.copy_from_slice(&mask.data);
            for y in 0..h {
                let y0 = y.saturating_sub(1);
                let y1 = (y + 1).min(h - 1);
                for x in 0..w {
                    let x0 = x.saturating_sub(1);
                    let x1 = (x + 1).min(w - 1);
                    let mut m = 0.0f32;
                    for ry in y0..=y1 {
                        let base = ry * w;
                        for rx in x0..=x1 {
                            m = m.max(scratch[base + rx]);
                        }
                    }
                    mask.data[y * w + x] = m;
                }
            }
        }
        feather_mask(mask, dilate);
    }
    if contrast > 0.0 {
        let (lo, hi) = ((contrast - 0.15).max(0.0), (contrast + 0.15).min(1.0));
        let span = (hi - lo).max(1e-6);
        for v in mask.data.iter_mut() {
            let t = ((*v - lo) / span).clamp(0.0, 1.0);
            *v = t * t * (3.0 - 2.0 * t); // smoothstep
        }
    }
    Ok(())
}

/// Box blur with radius `r` (kernel `(2r+1)^2`), clamp padding, sliding
/// window sums — the same separable scheme as the guided filter, cheap
/// enough to run per frame at 640x480 (~0.3 ms).
fn feather_mask(mask: &mut Mask, r: u32) {
    let w = mask.resolution.width as usize;
    let h = mask.resolution.height as usize;
    let r = r as i64;
    let n = w * h;
    let mut tmp = vec![0f32; n];
    // Horizontal pass.
    for y in 0..h {
        let base = y * w;
        let mut acc = 0.0f64;
        acc += (r + 1) as f64 * mask.data[base] as f64;
        for i in 1..=r {
            acc += mask.data[base + (i as usize).min(w - 1)] as f64;
        }
        tmp[base] = (acc / (2 * r + 1) as f64) as f32;
        for x in 1..w {
            acc -= mask.data[base + (x as i64 - 1 - r).clamp(0, w as i64 - 1) as usize] as f64;
            acc += mask.data[base + (x as i64 + r).min(w as i64 - 1) as usize] as f64;
            tmp[base + x] = (acc / (2 * r + 1) as f64) as f32;
        }
    }
    // Vertical pass.
    for x in 0..w {
        let mut acc = 0.0f64;
        acc += (r + 1) as f64 * tmp[x] as f64;
        for i in 1..=r {
            acc += tmp[((i as usize).min(h - 1)) * w + x] as f64;
        }
        mask.data[x] = (acc / (2 * r + 1) as f64) as f32;
        for y in 1..h {
            acc -= tmp[((y as i64 - 1 - r).clamp(0, h as i64 - 1)) as usize * w + x] as f64;
            acc += tmp[((y as i64 + r).min(h as i64 - 1)) as usize * w + x] as f64;
            mask.data[y * w + x] = (acc / (2 * r + 1) as f64) as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use frameiru_core::format::Resolution;

    fn res(w: u32, h: u32) -> Resolution {
        Resolution {
            width: w,
            height: h,
        }
    }

    #[test]
    fn dilation_fills_small_holes() {
        // 3x3, center = 0, ring = 1: one pass of 3x3 max fills the hole.
        let mut mask = Mask {
            resolution: res(3, 3),
            data: vec![
                1.0, 1.0, 1.0, //
                1.0, 0.0, 1.0, //
                1.0, 1.0, 1.0,
            ],
        };
        polish_mask(&mut mask, 1, 0.0).unwrap();
        assert_eq!(mask.data[4], 1.0, "hole must be filled");
        // Far-away zero (corner gap of 2 px) stays low through polish: a
        // 1px dilate cannot bridge it, and the feather only softens it.
        let mut mask = Mask {
            resolution: res(3, 3),
            data: vec![
                0.0, 0.0, 1.0, //
                0.0, 0.0, 1.0, //
                1.0, 1.0, 1.0,
            ],
        };
        polish_mask(&mut mask, 1, 0.0).unwrap();
        // The feather spans 3 rows, so the 2px corner dip softens but must
        // stay visibly dimmer than the solid corner.
        assert!(
            mask.data[0] < 0.66,
            "2px gap must stay background-ish: {}",
            mask.data[0]
        );
        assert!(mask.data[0] < mask.data[8], "corner dimmer than solid");
        assert!(
            mask.data[1] > 0.5,
            "diagonal neighbor fills corner-adjacent: {}",
            mask.data[1]
        );
    }

    #[test]
    fn dilation_is_idempotent_on_solid_masks() {
        let mut mask = Mask {
            resolution: res(4, 4),
            data: vec![0.7; 16],
        };
        polish_mask(&mut mask, 3, 0.0).unwrap();
        assert!(mask.data.iter().all(|&v| v == 0.7));
    }

    /// U9.8 regression: plain dilation max-filters the matte edge to 1.0
    /// ("blur never reaches the subject" ring); the feather must restore a
    /// soft transition band while keeping the enlarged coverage.
    #[test]
    fn feather_restores_soft_edge_after_dilation() {
        // 9x3: left 3 cols solid fg (1.0), right 6 cols bg (0.0).
        let mut mask = Mask {
            resolution: res(9, 3),
            data: vec![1.0; 27],
        };
        for y in 0..3usize {
            for x in 4..9usize {
                mask.data[y * 9 + x] = 0.0;
            }
        }
        // Dilate 1: the edge columns 2..4+ are all 1.0 (hard), then feather
        // (box 3) must produce strictly decreasing values across the edge.
        polish_mask(&mut mask, 1, 0.0).unwrap();
        let row = &mask.data[0..9];
        assert_eq!(row[0], 1.0, "subject interior stays solid");
        assert!(
            row[4] > 0.5,
            "dilated coverage still reaches col 4 (feathered): {}",
            row[4]
        );
        // The transition at the coverage boundary must be graded, not a
        // cliff: strict descent from row[4] down to the background plateau.
        assert!(
            row[4] > row[5] && row[5] > row[6] && row[6] >= row[7] && row[7] == 0.0,
            "feathered edge must grade down, got {row:?}"
        );
        // And the dilated + feathered boundary must cross 0.5 NO EARLIER
        // than the original boundary (coverage did not recede): original
        // step was between col 2/3; coverage landed at col 4+.
        let crossing = (0..9).find(|&x| row[x] < 0.5).unwrap_or(9);
        assert!(
            crossing >= 4,
            "coverage must not recede, cross at col {crossing}"
        );
    }

    #[test]
    fn feather_only_runs_with_dilation() {
        // Without dilation a soft mask stays untouched (no feather pass):
        // value at an edge pixel must survive exactly when contrast is off.
        let mut mask = Mask {
            resolution: res(3, 1),
            data: vec![0.5, 0.5, 0.0],
        };
        let before = mask.data.clone();
        polish_mask(&mut mask, 0, 0.0).unwrap();
        assert_eq!(mask.data, before);
    }

    #[test]
    fn contrast_collapses_mid_alphas() {
        let mut mask = Mask {
            resolution: res(5, 1),
            data: vec![0.0, 0.25, 0.5, 0.75, 1.0],
        };
        polish_mask(&mut mask, 0, 0.5).unwrap();
        // Band [0.35, 0.65]: 0.5 maps to exactly 0.5 (smoothstep center);
        // 0.25/0.75 collapse hard toward 0/1.
        assert!((mask.data[0] - 0.0).abs() < 1e-6);
        assert!(
            mask.data[1] < 0.05,
            "0.25 must collapse toward 0: {}",
            mask.data[1]
        );
        assert!((mask.data[2] - 0.5).abs() < 1e-6);
        assert!(
            mask.data[3] > 0.95,
            "0.75 must collapse toward 1: {}",
            mask.data[3]
        );
        assert!((mask.data[4] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn zero_config_is_noop() {
        let mut mask = Mask {
            resolution: res(2, 2),
            data: vec![0.1, 0.5, 0.9, 0.3],
        };
        let before = mask.data.clone();
        polish_mask(&mut mask, 0, 0.0).unwrap();
        assert_eq!(mask.data, before);
    }

    #[test]
    fn rejects_invalid_mask() {
        let mut mask = Mask {
            resolution: res(0, 0),
            data: Vec::new(),
        };
        assert!(polish_mask(&mut mask, 1, 0.4).is_err());
    }
}
