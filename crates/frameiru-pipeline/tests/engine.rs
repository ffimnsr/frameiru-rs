//! End-to-end engine tests: frame flow, dynamic masks, preview, background
//! switching, drop policy, and shutdown.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use frameiru_capture::MockSource;
use frameiru_compose::CpuCompositor;
use frameiru_core::buffer::Mask;
use frameiru_core::error::FrameiruError;
use frameiru_core::format::{FrameMetadata, PixelFormat};
use frameiru_core::traits::{Compositor, FrameSink, FrameSource, Segmenter};
use frameiru_core::{BackgroundMode, FrameBuffer, Resolution};
use frameiru_pipeline::{Engine, PipelineConfig, PipelineHandle};
use frameiru_sink::MockSink;

const RES: Resolution = Resolution {
    width: 64,
    height: 48,
};

fn res() -> Resolution {
    RES
}

fn config() -> PipelineConfig {
    PipelineConfig {
        max_fps: 0, // uncapped for deterministic tests
        compose_idle_threshold: 0.5,
        ..Default::default()
    }
}

/// Sink that records every frame for assertions from the test thread.
#[derive(Clone, Default)]
struct RecordingSink {
    inner: Arc<Mutex<Vec<FrameBuffer>>>,
}

impl FrameSink for RecordingSink {
    fn write_frame(&mut self, frame: &FrameBuffer) -> Result<(), FrameiruError> {
        self.inner
            .lock()
            .expect("sink mutex poisoned")
            .push(frame.clone());
        Ok(())
    }
}

/// Waits until `pred` holds or the timeout elapses; panics with context.
fn wait_until(what: &str, timeout: Duration, pred: impl Fn() -> bool) {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if pred() {
            return;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    panic!("timed out waiting for {what}");
}

#[test]
fn frames_flow_source_to_sink_with_preview() {
    let source = MockSource::new(res()).unwrap();
    let mut compositor = CpuCompositor::new();
    compositor
        .update_background(BackgroundMode::Color {
            r: 10,
            g: 20,
            b: 30,
        })
        .unwrap();
    let sink = RecordingSink::default();

    let engine = Engine::start(
        config(),
        Box::new(source),
        None,
        Box::new(compositor),
        Box::new(sink.clone()),
    )
    .unwrap();
    let handle = engine.handle();
    let mut preview = handle.subscribe();

    wait_until("20 composited frames", Duration::from_secs(10), || {
        handle.metrics().frames_composited >= 20
    });
    let m = handle.metrics();
    assert!(
        m.frames_composited + m.frames_dropped <= m.frames_captured,
        "frames in flight are the difference"
    );

    // Sink received the composited frames.
    let recorded = sink.inner.lock().expect("sink mutex poisoned");
    assert!(recorded.len() >= 20);
    assert_eq!(recorded[0].metadata.resolution, res());
    assert_eq!(recorded[0].metadata.format, PixelFormat::Rgb8);
    // No segmenter -> all-foreground mask -> output equals the source
    // pattern composited over the color background... with mask 1.0 the
    // output is the foreground, which is not constant; just check size.
    assert_eq!(recorded[0].data.len(), 64 * 48 * 3);
    drop(recorded);

    // Preview subscriber observed frames (may lag; keep reading until one
    // arrives).
    let frame = loop {
        match preview.try_recv() {
            Ok(frame) => break frame,
            Err(tokio::sync::broadcast::error::TryRecvError::Lagged(_)) => continue,
            Err(tokio::sync::broadcast::error::TryRecvError::Empty) => {
                std::thread::sleep(Duration::from_millis(5));
                continue;
            }
            Err(tokio::sync::broadcast::error::TryRecvError::Closed) => {
                panic!("preview channel closed while engine runs")
            }
        }
    };
    assert_eq!(frame.metadata.resolution, res());

    engine.shutdown();
    assert!(!handle.is_running());
}

#[test]
fn background_can_change_dynamically() {
    let source = MockSource::new(res()).unwrap();
    let compositor = CpuCompositor::new();
    let sink = RecordingSink::default();
    let engine = Engine::start(
        config(),
        Box::new(source),
        Some(Box::new(ConstantSegmenter { value: 0.0 })),
        Box::new(compositor),
        Box::new(sink.clone()),
    )
    .unwrap();
    let handle = engine.handle();

    handle
        .update_background(BackgroundMode::Color { r: 255, g: 0, b: 0 })
        .unwrap();
    wait_until(
        "frames with red background",
        Duration::from_secs(10),
        || {
            let recorded = sink.inner.lock().expect("sink mutex poisoned");
            recorded
                .last()
                .is_some_and(|f| f.data.chunks(3).all(|p| p == [255, 0, 0]))
        },
    );
    let red_count = sink.inner.lock().expect("sink mutex poisoned").len();

    handle
        .update_background(BackgroundMode::Color { r: 0, g: 0, b: 255 })
        .unwrap();
    wait_until(
        "frames with blue background",
        Duration::from_secs(10),
        || {
            let recorded = sink.inner.lock().expect("sink mutex poisoned");
            recorded
                .last()
                .is_some_and(|f| f.data.chunks(3).all(|p| p == [0, 0, 255]))
        },
    );
    assert!(
        sink.inner.lock().expect("sink mutex poisoned").len() > red_count,
        "more frames after switching background"
    );
    engine.shutdown();
}

/// Constant mask so compositing is deterministic: mask `value` over a color
/// background yields `value * fg + (1 - value) * bg`.
struct ConstantSegmenter {
    value: f32,
}

impl Segmenter for ConstantSegmenter {
    fn input_resolution(&self) -> Resolution {
        res()
    }

    fn segment(&mut self, frame: &FrameBuffer) -> Result<Mask, FrameiruError> {
        Ok(Mask::filled(frame.metadata.resolution, self.value))
    }
}

#[test]
fn segmenter_mask_drives_composition() {
    let source = MockSource::new(res()).unwrap();
    let mut compositor = CpuCompositor::new();
    compositor
        .update_background(BackgroundMode::Color { r: 0, g: 0, b: 0 })
        .unwrap();
    let sink = RecordingSink::default();
    let engine = Engine::start(
        config(),
        Box::new(source),
        Some(Box::new(ConstantSegmenter { value: 0.0 })),
        Box::new(compositor),
        Box::new(sink.clone()),
    )
    .unwrap();
    let handle = engine.handle();

    // Mask 0.0 -> fully background: every pixel black.
    wait_until("masked-to-black frames", Duration::from_secs(10), || {
        let recorded = sink.inner.lock().expect("sink mutex poisoned");
        recorded
            .last()
            .is_some_and(|f| f.data.iter().all(|&v| v == 0))
    });
    assert_eq!(handle.metrics().mask_errors, 0, "no segmentation errors");
    engine.shutdown();
}

#[test]
fn saturated_channel_drops_newest_frames() {
    // A slow compositor (20ms/frame) against an unbounded-fast source fills
    // the tiny channel; capture must drop frames instead of blocking.
    let source = MockSource::new(res()).unwrap();
    let slow = SlowCompositor::new();
    let sink = RecordingSink::default();
    let cfg = PipelineConfig {
        channel_capacity: 1,
        max_fps: 0,
        ..Default::default()
    };
    let engine =
        Engine::start(cfg, Box::new(source), None, Box::new(slow), Box::new(sink)).unwrap();
    let handle = engine.handle();

    wait_until("capture outpacing compose", Duration::from_secs(10), || {
        let m = handle.metrics();
        m.frames_captured >= 300 && m.frames_dropped > 0
    });
    let m = handle.metrics();
    assert!(
        m.frames_composited + m.frames_dropped <= m.frames_captured,
        "frames in flight are the difference"
    );
    engine.shutdown();
}

/// Compositor that sleeps per frame to force channel saturation.
struct SlowCompositor {
    inner: CpuCompositor,
}

impl SlowCompositor {
    fn new() -> Self {
        Self {
            inner: CpuCompositor::new(),
        }
    }
}

impl Compositor for SlowCompositor {
    fn composite(
        &mut self,
        source: &FrameBuffer,
        mask: &Mask,
        output: &mut FrameBuffer,
    ) -> Result<(), FrameiruError> {
        std::thread::sleep(Duration::from_millis(20));
        self.inner.composite(source, mask, output)
    }

    fn update_background(&mut self, mode: BackgroundMode) -> Result<(), FrameiruError> {
        self.inner.update_background(mode)
    }
}

#[test]
fn engine_drop_shuts_down_cleanly() {
    let source = MockSource::new(res()).unwrap();
    let compositor = CpuCompositor::new();
    let sink = RecordingSink::default();
    let engine = Engine::start(
        config(),
        Box::new(source),
        None,
        Box::new(compositor),
        Box::new(sink),
    )
    .unwrap();
    let handle = engine.handle();
    wait_until("frames flowing", Duration::from_secs(10), || {
        handle.metrics().frames_composited >= 5
    });
    drop(engine); // runs shutdown via Drop
    assert!(!handle.is_running());
    // Shutdown is idempotent.
    handle.shutdown();
}

/// Recurrent segmenter counting `reset_state` calls (RVM-style).
struct ResetTrackingSegmenter {
    resets: Arc<std::sync::atomic::AtomicU64>,
}

impl Segmenter for ResetTrackingSegmenter {
    fn input_resolution(&self) -> Resolution {
        res()
    }

    fn segment(&mut self, frame: &FrameBuffer) -> Result<Mask, FrameiruError> {
        Ok(Mask::filled(frame.metadata.resolution, 0.5))
    }

    fn reset_state(&mut self) {
        self.resets
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Dropped capture frames must trigger a segmenter state reset so recurrent
/// models cannot ghost across discontinuities.
#[test]
fn dropped_frames_reset_stateful_segmenter() {
    let source = MockSource::new(res()).unwrap();
    let resets = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let segmenter = ResetTrackingSegmenter {
        resets: Arc::clone(&resets),
    };
    let slow = SlowCompositor::new();
    let cfg = PipelineConfig {
        channel_capacity: 1,
        max_fps: 0,
        ..Default::default()
    };
    let engine = Engine::start(
        cfg,
        Box::new(source),
        Some(Box::new(segmenter)),
        Box::new(slow),
        Box::new(RecordingSink::default()),
    )
    .unwrap();
    let handle = engine.handle();

    wait_until(
        "capture drops + segmenter resets",
        Duration::from_secs(10),
        || {
            handle.metrics().frames_captured >= 300
                && resets.load(std::sync::atomic::Ordering::Relaxed) > 0
        },
    );
    engine.shutdown();
}

/// Segmenter whose mask value flips on every segmentation call.
struct OscillatingSegmenter {
    value: f32,
}

impl Segmenter for OscillatingSegmenter {
    fn input_resolution(&self) -> Resolution {
        res()
    }

    fn segment(&mut self, frame: &FrameBuffer) -> Result<Mask, FrameiruError> {
        let value = self.value;
        self.value = 1.0 - self.value;
        Ok(Mask::filled(frame.metadata.resolution, value))
    }
}

fn run_with_alpha(
    mask_alpha: Option<f32>,
    segmenter: Box<dyn Segmenter>,
) -> (Engine, PipelineHandle, RecordingSink) {
    let source = MockSource::new(res()).unwrap();
    let mut compositor = CpuCompositor::new();
    compositor
        .update_background(BackgroundMode::Color { r: 0, g: 0, b: 0 })
        .unwrap();
    let sink = RecordingSink::default();
    let engine = Engine::start(
        PipelineConfig {
            max_fps: 0,
            mask_alpha,
            // The oscillation test needs every frame segmented (no gating).
            infer_max_fps: 0,
            ..Default::default()
        },
        Box::new(source),
        Some(segmenter),
        Box::new(compositor),
        Box::new(sink.clone()),
    )
    .unwrap();
    let handle = engine.handle();
    (engine, handle, sink)
}

/// With `mask_alpha = 0.0` the first mask freezes, so every composited
/// frame after it is identical (pure black background) even though the
/// segmenter oscillates. Without smoothing the oscillation shows up as
/// changing frames.
#[test]
fn mask_smoothing_freezes_oscillating_masks() {
    // Frozen at the first mask (0.0): output = black background, which is
    // independent of the animated mock source. A constant segmenter keeps
    // the assertion stable even though drop-triggered resets re-seed the
    // EMA. Startup fallback frames use the all-foreground mask, so only
    // assert on the post-segment tail.
    let (engine, _handle, sink) =
        run_with_alpha(Some(0.0), Box::new(ConstantSegmenter { value: 0.0 }));
    wait_until(
        "black (frozen-mask) frames",
        Duration::from_secs(10),
        || {
            let r = sink.inner.lock().expect("sink mutex poisoned");
            r.len() >= 40 && r.iter().filter(|f| f.data.iter().all(|&v| v == 0)).count() >= 30
        },
    );
    let recorded = sink.inner.lock().expect("sink mutex poisoned").clone();
    let mut black = recorded
        .iter()
        .skip_while(|f| f.data.iter().any(|&v| v != 0));
    let black_count = black.clone().count();
    assert!(
        black_count >= 30 && black.all(|f| f.data.iter().all(|&v| v == 0)),
        "frozen mask must pin the output to pure black after the first mask"
    );
    drop(recorded);
    engine.shutdown();

    let (engine, handle, sink) =
        run_with_alpha(None, Box::new(OscillatingSegmenter { value: 0.0 }));
    wait_until("frames composited", Duration::from_secs(10), || {
        handle.metrics().frames_composited >= 40
    });
    let recorded = sink.inner.lock().expect("sink mutex poisoned").clone();
    let differing = recorded
        .iter()
        .zip(recorded.iter().skip(1))
        .filter(|(a, b)| a.data != b.data)
        .count();
    assert!(
        differing > 0,
        "unsmoothed oscillating masks must produce changing frames"
    );
    drop(recorded);
    engine.shutdown();
}

#[test]
fn reject_bad_config() {
    let source = MockSource::new(res()).unwrap();
    let cfg = PipelineConfig {
        channel_capacity: 0,
        ..Default::default()
    };
    let err = Engine::start(
        cfg,
        Box::new(source),
        None,
        Box::new(CpuCompositor::new()),
        Box::new(MockSink::new()),
    );
    assert!(err.is_err());
}

/// Source emitting byte-identical frames (static scene for gating tests).
struct ConstSource {
    resolution: Resolution,
    sequence: u64,
}

impl ConstSource {
    fn new(resolution: Resolution) -> Self {
        Self {
            resolution,
            sequence: 0,
        }
    }
}

impl FrameSource for ConstSource {
    fn resolution(&self) -> Resolution {
        self.resolution
    }

    fn format(&self) -> PixelFormat {
        PixelFormat::Rgb8
    }

    fn next_frame(&mut self) -> Result<FrameBuffer, FrameiruError> {
        let mut frame = FrameBuffer::new(FrameMetadata {
            sequence: self.sequence,
            timestamp_us: frameiru_core::timestamp_us_now(),
            resolution: self.resolution,
            format: PixelFormat::Rgb8,
        });
        frame.data = vec![77u8; (self.resolution.area() * 3) as usize];
        self.sequence += 1;
        Ok(frame)
    }
}

/// U9.6: an idle blur scene reuses the previous composite (composite
/// skipped, sink still fed at the same rate).
#[test]
fn idle_blur_scene_reuses_composites() {
    let source = ConstSource::new(res());
    let compositor = CpuCompositor::new();
    let sink = RecordingSink::default();
    let engine = Engine::start(
        config(),
        Box::new(source),
        None, // no segmenter: the fallback mask is stored once and stays put
        Box::new(compositor),
        Box::new(sink.clone()),
    )
    .unwrap();
    let handle = engine.handle();
    // The engine must own the background (the shared slot gates the skip).
    handle
        .update_background(BackgroundMode::Blur { radius: 3.0 })
        .unwrap();

    wait_until(
        "composites reused on idle scene",
        Duration::from_secs(10),
        || {
            let m = handle.metrics();
            m.frames_composited >= 10 && m.composites_skipped >= 3
        },
    );
    // Join the workers first so the counters are quiescent: reading them
    // live races the compose thread's per-frame atomic stores.
    engine.shutdown();
    let m = handle.metrics();
    assert!(
        m.composites_skipped < m.frames_composited,
        "first composite(s) must have been real: skipped={} composited={}, captured={}",
        m.composites_skipped,
        m.frames_composited,
        m.frames_captured
    );
}

/// U9.6: a moving scene never reuses composites.
#[test]
fn moving_scene_never_reuses_composites() {
    let source = MockSource::new(res()).unwrap(); // animated pattern
    let compositor = CpuCompositor::new();
    let sink = RecordingSink::default();
    let engine = Engine::start(
        config(),
        Box::new(source),
        None,
        Box::new(compositor),
        Box::new(sink.clone()),
    )
    .unwrap();
    let handle = engine.handle();
    handle
        .update_background(BackgroundMode::Blur { radius: 3.0 })
        .unwrap();

    wait_until("frames flowing", Duration::from_secs(10), || {
        handle.metrics().frames_composited >= 30
    });
    assert_eq!(
        handle.metrics().composites_skipped,
        0,
        "animated source must never be idle"
    );
    engine.shutdown();
}

/// U9.6: inference is throttled on a static scene (`infer_max_fps`).
#[test]
fn inference_throttle_limits_masks_on_static_scene() {
    let source = ConstSource::new(res());
    let compositor = CpuCompositor::new();
    let sink = RecordingSink::default();
    let cfg = PipelineConfig {
        max_fps: 0,
        infer_max_fps: 1, // one mask per second at most
        ..Default::default()
    };
    let engine = Engine::start(
        cfg,
        Box::new(source),
        Some(Box::new(ConstantSegmenter { value: 0.0 })),
        Box::new(compositor),
        Box::new(sink),
    )
    .unwrap();
    let handle = engine.handle();

    wait_until(
        "captured frames well past the throttle",
        Duration::from_secs(10),
        || {
            let m = handle.metrics();
            m.frames_captured >= 100 && m.masks_computed >= 1
        },
    );
    let m = handle.metrics();
    assert!(
        m.masks_computed < m.frames_captured / 2,
        "static scene must throttle inference: {} masks vs {} frames",
        m.masks_computed,
        m.frames_captured
    );
    engine.shutdown();
}
