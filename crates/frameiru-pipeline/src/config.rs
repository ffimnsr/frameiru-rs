//! Engine configuration.

use frameiru_core::error::FrameiruError;

/// Tuning knobs for the pipeline engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelineConfig {
    /// Capacity of the internal frame channels (capture -> infer/compose).
    ///
    /// Frames that do not fit are dropped (newest dropped), which keeps
    /// latency bounded when a stage falls behind.
    pub channel_capacity: usize,
    /// Ring-buffer capacity of the preview broadcast stream.
    pub preview_capacity: usize,
    /// Cap on composited frames per second; `0` disables pacing.
    pub max_fps: u32,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            channel_capacity: 2,
            preview_capacity: 4,
            max_fps: 30,
        }
    }
}

impl PipelineConfig {
    pub fn validate(&self) -> Result<(), FrameiruError> {
        if self.channel_capacity == 0 {
            return Err(FrameiruError::InvalidArgument(
                "channel_capacity must be at least 1".into(),
            ));
        }
        if self.preview_capacity == 0 {
            return Err(FrameiruError::InvalidArgument(
                "preview_capacity must be at least 1".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_valid() {
        let cfg = PipelineConfig::default();
        assert!(cfg.validate().is_ok());
        assert_eq!(cfg.channel_capacity, 2);
        assert_eq!(cfg.preview_capacity, 4);
        assert_eq!(cfg.max_fps, 30);
    }

    #[test]
    fn rejects_zero_capacities() {
        let bad = PipelineConfig {
            channel_capacity: 0,
            ..Default::default()
        };
        assert!(bad.validate().is_err());
        let bad = PipelineConfig {
            preview_capacity: 0,
            ..Default::default()
        };
        assert!(bad.validate().is_err());
    }
}
