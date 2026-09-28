//! Exponential moving average (EMA) filter for segmentation masks.
//!
//! Raw model output flickers between frames; blending each new mask with the
//! previous one smooths edges and suppresses single-frame errors.

use frameiru_core::buffer::Mask;
use frameiru_core::error::FrameiruError;

/// Blends `alpha * new + (1 - alpha) * previous`.
///
/// `alpha = 1.0` disables smoothing (passthrough); `alpha = 0.0` freezes the
/// first mask. The first update initializes the state and returns the input
/// unchanged.
#[derive(Debug, Clone)]
pub struct TemporalSmoother {
    alpha: f32,
    state: Option<Mask>,
}

impl TemporalSmoother {
    /// Creates a smoother with blending factor `alpha` in `[0, 1]`.
    pub fn new(alpha: f32) -> Result<Self, FrameiruError> {
        if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
            return Err(FrameiruError::InvalidArgument(format!(
                "smoother alpha must be in [0, 1], got {alpha}"
            )));
        }
        Ok(Self { alpha, state: None })
    }

    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// Replaces the blending factor; keeps the current state.
    pub fn set_alpha(&mut self, alpha: f32) -> Result<(), FrameiruError> {
        if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
            return Err(FrameiruError::InvalidArgument(format!(
                "smoother alpha must be in [0, 1], got {alpha}"
            )));
        }
        self.alpha = alpha;
        Ok(())
    }

    /// Blends `mask` into the running state and returns the smoothed mask.
    ///
    /// If `mask` does not match the stored state's resolution, the state is
    /// reset and `mask` is returned unchanged (e.g. after a camera
    /// resolution change).
    pub fn update(&mut self, mask: &Mask) -> Mask {
        match &self.state {
            None => {
                self.state = Some(mask.clone());
                mask.clone()
            }
            Some(prev)
                if prev.resolution != mask.resolution || prev.data.len() != mask.data.len() =>
            {
                self.reset();
                self.state = Some(mask.clone());
                mask.clone()
            }
            Some(prev) => {
                let mut out = Mask {
                    resolution: mask.resolution,
                    data: Vec::with_capacity(mask.data.len()),
                };
                let (a, b) = (self.alpha, 1.0 - self.alpha);
                for (new, old) in mask.data.iter().zip(prev.data.iter()) {
                    out.data.push(a * new + b * old);
                }
                self.state = Some(out.clone());
                out
            }
        }
    }

    /// Drops the running state; the next update starts fresh.
    pub fn reset(&mut self) {
        self.state = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use frameiru_core::format::Resolution;

    fn mask(values: &[f32]) -> Mask {
        Mask {
            resolution: Resolution {
                width: values.len() as u32,
                height: 1,
            },
            data: values.to_vec(),
        }
    }

    #[test]
    fn first_update_returns_input_unchanged() {
        let mut s = TemporalSmoother::new(0.3).unwrap();
        let m = mask(&[0.1, 0.9]);
        assert_eq!(s.update(&m).data, [0.1, 0.9]);
        assert_eq!(s.alpha(), 0.3);
    }

    #[test]
    fn alpha_one_is_passthrough() {
        let mut s = TemporalSmoother::new(1.0).unwrap();
        s.update(&mask(&[0.5]));
        assert_eq!(s.update(&mask(&[0.8])).data, [0.8]);
    }

    #[test]
    fn alpha_zero_holds_first_mask() {
        let mut s = TemporalSmoother::new(0.0).unwrap();
        s.update(&mask(&[0.2]));
        assert_eq!(s.update(&mask(&[0.9])).data, [0.2]);
    }

    #[test]
    fn exponential_blend_formula() {
        let mut s = TemporalSmoother::new(0.25).unwrap();
        s.update(&mask(&[0.5, 1.0]));
        // 0.25 * 1.0 + 0.75 * 0.5 = 0.625; 0.25 * 0.0 + 0.75 * 1.0 = 0.75.
        assert_eq!(s.update(&mask(&[1.0, 0.0])).data, [0.625, 0.75]);
    }

    #[test]
    fn converges_to_constant_input() {
        let mut s = TemporalSmoother::new(0.5).unwrap();
        s.update(&mask(&[0.0]));
        let mut last = 0.0;
        for _ in 0..20 {
            last = s.update(&mask(&[1.0])).data[0];
        }
        // 1 - 0.5^21 ~ 1.0 to f32 precision.
        assert!((last - 1.0).abs() < 1e-5);
    }

    #[test]
    fn reset_restarts_state() {
        let mut s = TemporalSmoother::new(0.5).unwrap();
        s.update(&mask(&[1.0]));
        s.reset();
        assert_eq!(s.update(&mask(&[0.1])).data, [0.1]);
    }

    #[test]
    fn resolution_change_resets_state() {
        let mut s = TemporalSmoother::new(0.5).unwrap();
        s.update(&mask(&[1.0, 0.0]));
        // Wider mask: mismatched state, must return input unchanged.
        let wider = Mask {
            resolution: Resolution {
                width: 2,
                height: 2,
            },
            data: vec![0.7; 4],
        };
        assert_eq!(s.update(&wider).data, vec![0.7; 4]);
        assert_eq!(s.update(&mask(&[0.5, 0.5])).data, [0.5, 0.5]);
    }

    #[test]
    fn rejects_out_of_range_alpha() {
        assert!(TemporalSmoother::new(-0.1).is_err());
        assert!(TemporalSmoother::new(1.1).is_err());
        assert!(TemporalSmoother::new(f32::NAN).is_err());
        let mut s = TemporalSmoother::new(0.5).unwrap();
        assert!(s.set_alpha(2.0).is_err());
        assert_eq!(s.alpha(), 0.5, "failed set_alpha must not change alpha");
    }
}
