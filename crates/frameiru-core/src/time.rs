//! Wall-clock timestamps shared by capture sources and the pipeline.

use std::time::{SystemTime, UNIX_EPOCH};

/// Current time in microseconds since the Unix epoch.
///
/// Every source stamps frames with this clock so the pipeline's latency
/// metric (capture timestamp vs. compose time) is meaningful.
pub fn timestamp_us_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestamps_are_monotonic_in_practice() {
        let a = timestamp_us_now();
        let b = timestamp_us_now();
        assert!(b >= a);
        // Sanity: epoch micros are a huge, plausible number.
        assert!(a > 1_000_000_000_000);
    }
}
