//! End-to-end engine tests: frame flow, dynamic masks, preview, background
//! switching, drop policy, and shutdown.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use frameiru_capture::MockSource;
use frameiru_compose::CpuCompositor;
use frameiru_core::buffer::Mask;
use frameiru_core::error::FrameiruError;
use frameiru_core::format::PixelFormat;
use frameiru_core::traits::{Compositor, FrameSink, Segmenter};
use frameiru_core::{BackgroundMode, FrameBuffer, Resolution};
use frameiru_pipeline::{Engine, PipelineConfig};
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
