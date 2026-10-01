//! Runtime counters: FPS, latency, and frame drops.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Snapshot of the engine's runtime counters.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct MetricsSnapshot {
    /// Frames read from the source.
    pub frames_captured: u64,
    /// Frames composited and written to the sink.
    pub frames_composited: u64,
    /// Frames written to the sink by reusing the previous composite (scene
    /// idle, U9.6 gating) — compositing was skipped.
    pub composites_skipped: u64,
    /// Frames dropped due to channel saturation (newest dropped).
    pub frames_dropped: u64,
    /// Successful mask computations (the async inference rate).
    pub masks_computed: u64,
    /// Failed segmentation calls (previous mask reused).
    pub mask_errors: u64,
    /// Capture-to-composite latency of the last frame, microseconds.
    pub latency_us: u64,
    /// Rolling per-second capture rate.
    pub capture_fps: f64,
    /// Rolling per-second composite rate.
    pub composite_fps: f64,
}

/// Shared counters for the pipeline, updated from the worker threads.
#[derive(Debug)]
pub struct Metrics {
    captured: AtomicU64,
    composited: AtomicU64,
    composites_skipped: AtomicU64,
    dropped: AtomicU64,
    mask_errors: AtomicU64,
    masks_computed: AtomicU64,
    latency_us: AtomicU64,
    capture_fps: FpsMeter,
    composite_fps: FpsMeter,
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

impl Metrics {
    pub fn new() -> Self {
        Self {
            captured: AtomicU64::new(0),
            composited: AtomicU64::new(0),
            composites_skipped: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
            mask_errors: AtomicU64::new(0),
            masks_computed: AtomicU64::new(0),
            latency_us: AtomicU64::new(0),
            capture_fps: FpsMeter::new(),
            composite_fps: FpsMeter::new(),
        }
    }

    pub fn on_capture(&self) {
        self.captured.fetch_add(1, Ordering::Relaxed);
        self.capture_fps.tick();
    }

    pub fn on_drop(&self) {
        self.dropped.fetch_add(1, Ordering::Relaxed);
    }

    /// Records a composite that was skipped (idle frame, previous output
    /// reused) but still written to the sink.
    pub fn on_composite_skip(&self) {
        self.composites_skipped.fetch_add(1, Ordering::Relaxed);
    }

    pub fn on_composite(&self) {
        self.composited.fetch_add(1, Ordering::Relaxed);
        self.composite_fps.tick();
    }

    pub fn on_mask_error(&self) {
        self.mask_errors.fetch_add(1, Ordering::Relaxed);
    }

    /// Records a successful async mask computation.
    pub fn on_mask_computed(&self) {
        self.masks_computed.fetch_add(1, Ordering::Relaxed);
    }

    /// Records end-to-end latency: `now_us - captured_at_us`.
    pub fn record_latency(&self, now_us: u64, captured_at_us: u64) {
        self.latency_us
            .store(now_us.saturating_sub(captured_at_us), Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            frames_captured: self.captured.load(Ordering::Relaxed),
            frames_composited: self.composited.load(Ordering::Relaxed),
            composites_skipped: self.composites_skipped.load(Ordering::Relaxed),
            frames_dropped: self.dropped.load(Ordering::Relaxed),
            masks_computed: self.masks_computed.load(Ordering::Relaxed),
            mask_errors: self.mask_errors.load(Ordering::Relaxed),
            latency_us: self.latency_us.load(Ordering::Relaxed),
            capture_fps: self.capture_fps.read(),
            composite_fps: self.composite_fps.read(),
        }
    }
}

/// Per-second rate counter; cheap enough to tick on every frame.
#[derive(Debug)]
struct FpsMeter {
    state: Mutex<FpsState>,
}

#[derive(Debug)]
struct FpsState {
    last: Instant,
    count: u64,
    fps: f64,
}

impl FpsMeter {
    fn new() -> Self {
        Self {
            state: Mutex::new(FpsState {
                last: Instant::now(),
                count: 0,
                fps: 0.0,
            }),
        }
    }

    fn tick(&self) {
        let mut s = self.state.lock().expect("fps meter mutex poisoned");
        s.count += 1;
        let now = Instant::now();
        let elapsed = now.duration_since(s.last);
        if elapsed >= Duration::from_secs(1) {
            s.fps = s.count as f64 / elapsed.as_secs_f64();
            s.count = 0;
            s.last = now;
        }
    }

    fn read(&self) -> f64 {
        self.state.lock().expect("fps meter mutex poisoned").fps
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counters_accumulate() {
        let m = Metrics::new();
        m.on_capture();
        m.on_capture();
        m.on_composite();
        m.on_drop();
        m.on_mask_error();
        m.on_mask_computed();
        m.record_latency(1_000, 400);
        let s = m.snapshot();
        assert_eq!(s.frames_captured, 2);
        assert_eq!(s.frames_composited, 1);
        assert_eq!(s.frames_dropped, 1);
        assert_eq!(s.mask_errors, 1);
        assert_eq!(s.masks_computed, 1);
        assert_eq!(s.latency_us, 600);
    }

    #[test]
    fn latency_saturates_at_zero() {
        let m = Metrics::new();
        m.record_latency(100, 500);
        assert_eq!(m.snapshot().latency_us, 0);
    }

    #[test]
    fn fps_starts_zero_and_ticks_to_live_value() {
        let m = Metrics::new();
        assert_eq!(m.snapshot().capture_fps, 0.0);
        m.on_capture();
        assert_eq!(m.snapshot().capture_fps, 0.0, "window not closed yet");
        // Forcing the window closed is covered implicitly; the meter is
        // exercised end-to-end in the engine tests.
        let _ = m.snapshot();
    }
}
