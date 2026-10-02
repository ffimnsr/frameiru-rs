//! Embeddable pipeline engine and its cloneable control handle.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use arc_swap::ArcSwap;
use crossbeam_channel::bounded;
use frameiru_core::buffer::Mask;
use frameiru_core::error::FrameiruError;
use frameiru_core::traits::{Compositor, FrameSink, FrameSource, Segmenter};
use frameiru_core::{BackgroundMode, FrameBuffer, Resolution};
use tokio::sync::broadcast;

use crate::config::PipelineConfig;
use crate::metrics::MetricsSnapshot;
use crate::runner::{spawn_capture, spawn_compose, spawn_inference, PipelineShared};

/// A running multi-threaded pipeline.
///
/// Owns the capture/inference/composition workers; dropping the engine (or
/// calling [`Engine::shutdown`]) stops them cleanly. [`Engine::handle`]
/// returns a cheap cloneable [`PipelineHandle`] for control-plane use.
pub struct Engine {
    handle: PipelineHandle,
}

impl Engine {
    /// Starts the pipeline with the given stages.
    ///
    /// `source` and `sink` are required; `segmenter` is optional (without it
    /// the capture channel feeds the composer directly and the mask stays
    /// all-foreground).
    pub fn start(
        config: PipelineConfig,
        source: Box<dyn FrameSource>,
        segmenter: Option<Box<dyn Segmenter>>,
        compositor: Box<dyn Compositor>,
        sink: Box<dyn FrameSink>,
    ) -> Result<Self, FrameiruError> {
        config.validate()?;
        let subject_light = config.subject_light;

        let (capture_tx, capture_rx) = bounded::<FrameBuffer>(config.channel_capacity);
        let (infer_tx, infer_rx) = bounded::<FrameBuffer>(config.channel_capacity.max(4));
        let (mode_tx, mode_rx) = bounded::<BackgroundMode>(1);
        let (preview_tx, _) = broadcast::channel::<Arc<FrameBuffer>>(config.preview_capacity);
        let stop = Arc::new(AtomicBool::new(false));

        let shared = Arc::new(PipelineShared {
            config,
            metrics: crate::metrics::Metrics::new(),
            stop: Arc::clone(&stop),
            mask_slot: ArcSwap::from_pointee(Mask::default()),
            background: ArcSwap::from_pointee(BackgroundMode::Passthrough),
            last_resolution: ArcSwap::from_pointee(Resolution::default()),
            preview_tx,
            mode_tx,
            mode_rx,
            capture_tx,
            capture_rx,
            infer_tx,
            infer_rx,
        });

        let handle = PipelineHandle {
            inner: Arc::new(HandleInner {
                shared: Arc::clone(&shared),
                threads: Mutex::new(Vec::new()),
            }),
        };

        // Worker threads push their join handles into the handle so that
        // any clone (or the Engine itself) can join them on shutdown.
        let mut threads = handle.inner.threads.lock().expect("threads mutex poisoned");
        threads.push(spawn_capture(Arc::clone(&shared), source));

        let mut compositor = compositor;
        compositor.set_subject_light(subject_light);
        // Inference is optional and fully decoupled: it samples frames from
        // the capture stream into the mask slot without gating the video.
        if let Some(segmenter) = segmenter {
            threads.push(spawn_inference(
                Arc::clone(&shared),
                segmenter,
                shared.infer_rx.clone(),
            ));
        }
        threads.push(spawn_compose(
            Arc::clone(&shared),
            compositor,
            sink,
            shared.capture_rx.clone(),
        ));
        drop(threads);

        Ok(Self { handle })
    }

    /// Cloneable control plane for this engine.
    pub fn handle(&self) -> PipelineHandle {
        self.handle.clone()
    }

    /// Stops the workers and joins the threads. Idempotent; also runs on
    /// drop.
    pub fn shutdown(self) {
        self.handle.shutdown();
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        self.handle.shutdown();
    }
}

/// Shared control-plane state; cloned cheaply via `Arc`.
#[derive(Clone)]
pub struct PipelineHandle {
    inner: Arc<HandleInner>,
}

struct HandleInner {
    shared: Arc<PipelineShared>,
    threads: Mutex<Vec<JoinHandle<()>>>,
}

impl PipelineHandle {
    /// Subscribes to the preview stream. Every composited frame is shared
    /// with all subscribers as an `Arc`; lagging subscribers skip frames.
    pub fn subscribe(&self) -> broadcast::Receiver<Arc<FrameBuffer>> {
        self.inner.shared.preview_tx.subscribe()
    }

    /// Dynamically changes the background of the running pipeline.
    pub fn update_background(&self, mode: BackgroundMode) -> Result<(), FrameiruError> {
        self.inner
            .shared
            .mode_tx
            .send(mode)
            .map_err(|_| FrameiruError::Internal("pipeline is shutting down".into()))
    }

    /// Current runtime counters.
    pub fn metrics(&self) -> MetricsSnapshot {
        self.inner.shared.metrics.snapshot()
    }

    /// Whether the workers have been asked to stop.
    pub fn is_running(&self) -> bool {
        !self.inner.shared.stop.load(Ordering::Relaxed)
    }

    /// Background currently applied by the composer.
    pub fn current_background(&self) -> BackgroundMode {
        (**self.inner.shared.background.load()).clone()
    }

    /// Resolution of the last composited frame, if any.
    pub fn last_resolution(&self) -> Option<Resolution> {
        let res = **self.inner.shared.last_resolution.load();
        res.is_valid().then_some(res)
    }

    /// Signals all workers to stop and joins their threads. Idempotent.
    pub fn shutdown(&self) {
        self.inner.shared.stop.store(true, Ordering::Relaxed);
        let mut threads = self.inner.threads.lock().expect("threads mutex poisoned");
        for t in threads.drain(..) {
            if let Err(e) = t.join() {
                tracing::warn!("worker thread panicked: {e:?}");
            }
        }
    }
}

impl frameiru_ipc::Control for PipelineHandle {
    fn set_background(&self, mode: BackgroundMode) -> Result<(), FrameiruError> {
        self.update_background(mode)
    }

    fn status(&self) -> frameiru_ipc::StatusInfo {
        let m = self.metrics();
        frameiru_ipc::StatusInfo {
            running: self.is_running(),
            resolution: self.last_resolution(),
            capture_fps: m.capture_fps,
            composite_fps: m.composite_fps,
            frames_composited: m.frames_composited,
            composites_skipped: m.composites_skipped,
            masks_computed: m.masks_computed,
            latency_us: m.latency_us,
            background: self.current_background(),
        }
    }

    fn stop(&self) -> Result<(), FrameiruError> {
        self.shutdown();
        Ok(())
    }
}
