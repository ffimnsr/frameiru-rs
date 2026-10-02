//! Worker threads: capture, inference, and composition.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use arc_swap::ArcSwap;
use crossbeam_channel::{Receiver, TrySendError};
use frameiru_core::buffer::Mask;
use frameiru_core::traits::{Compositor, FrameSink, FrameSource, Segmenter};
use frameiru_core::{BackgroundMode, FrameBuffer, Resolution};
use tokio::sync::broadcast;

use crate::config::PipelineConfig;
use crate::metrics::Metrics;

/// Shared wiring handed to every worker thread.
pub(crate) struct PipelineShared {
    pub config: PipelineConfig,
    pub metrics: Metrics,
    pub stop: Arc<AtomicBool>,
    /// Dynamic mask slot: inference swaps it, compose reads the latest.
    pub mask_slot: ArcSwap<Mask>,
    /// Preview fan-out for UI consumers.
    pub preview_tx: broadcast::Sender<Arc<FrameBuffer>>,
    /// Background-mode update channel consumed by the composer.
    pub mode_tx: crossbeam_channel::Sender<BackgroundMode>,
    pub mode_rx: crossbeam_channel::Receiver<BackgroundMode>,
    /// Last applied background (for status queries).
    pub background: ArcSwap<BackgroundMode>,
    /// Resolution of the last composited frame (for status queries).
    pub last_resolution: ArcSwap<Resolution>,
    /// Capture -> compose channel. Saturation drops the newest frame.
    pub capture_tx: crossbeam_channel::Sender<FrameBuffer>,
    pub capture_rx: crossbeam_channel::Receiver<FrameBuffer>,
    /// Capture -> inference channel (unused without a segmenter). Inference
    /// is decoupled: it consumes frames at its own pace and only updates the
    /// mask slot, so a slow model never throttles the video.
    pub infer_tx: crossbeam_channel::Sender<FrameBuffer>,
    pub infer_rx: crossbeam_channel::Receiver<FrameBuffer>,
}

/// Spawns the capture worker. Frames that do not fit the channel are dropped
/// (newest dropped) so latency stays bounded when consumers fall behind.
pub(crate) fn spawn_capture(
    shared: Arc<PipelineShared>,
    mut source: Box<dyn FrameSource>,
) -> JoinHandle<()> {
    std::thread::Builder::new()
        .name("frameiru-capture".into())
        .spawn(move || {
            let mut warned = false;
            while !shared.stop.load(Ordering::Relaxed) {
                match source.next_frame() {
                    Ok(frame) => {
                        warned = false;
                        shared.metrics.on_capture();
                        // The compose channel gates the video; inference (if
                        // present) gets a clone and runs at its own pace. A
                        // full infer channel just skips a mask sample.
                        match shared.capture_tx.try_send(frame.clone()) {
                            Ok(()) => {}
                            Err(TrySendError::Full(_)) => shared.metrics.on_drop(),
                            Err(TrySendError::Disconnected(_)) => break,
                        }
                        let _ = shared.infer_tx.try_send(frame);
                    }
                    Err(e) => {
                        if !warned {
                            tracing::warn!("capture error: {e}");
                            warned = true;
                        }
                        // The source may block in a driver call; poll the
                        // stop flag so shutdown stays responsive.
                        for _ in 0..10 {
                            if shared.stop.load(Ordering::Relaxed) {
                                break;
                            }
                            std::thread::sleep(Duration::from_millis(10));
                        }
                    }
                }
            }
        })
        .expect("spawn capture thread")
}

/// Spawns the inference worker (only when a segmenter is configured): each
/// frame updates the dynamic mask slot. A failed segmentation keeps the
/// previous mask. Never blocks the video path.
pub(crate) fn spawn_inference(
    shared: Arc<PipelineShared>,
    segmenter: Box<dyn Segmenter>,
    input_rx: Receiver<FrameBuffer>,
) -> JoinHandle<()> {
    std::thread::Builder::new()
        .name("frameiru-inference".into())
        .spawn(move || {
            let mut segmenter = segmenter;
            let mut smoother = shared.config.mask_alpha.map(|alpha| {
                frameiru_segment::TemporalSmoother::new(alpha)
                    .expect("mask_alpha validated by PipelineConfig")
            });
            // Motion-gated cadence: static scenes skip segmentation, motion
            // overrides the cap (U9.6).
            let mut gate = crate::motion::InferGate::new(
                shared.config.infer_max_fps,
                shared.config.infer_motion_threshold,
            );
            let mut last_drops = 0u64;
            while !shared.stop.load(Ordering::Relaxed) {
                match input_rx.recv_timeout(STOP_POLL) {
                    Ok(mut frame) => {
                        // Drain any newer frames queued while inference was busy:
                        // inference must always operate on the most recently captured frame.
                        while let Ok(newer) = input_rx.try_recv() {
                            frame = newer;
                        }
                        // Dropped capture frames mean the video skipped a
                        // discontinuity: recurrent models and the EMA must
                        // reset so neither can ghost across it.
                        let drops = shared.metrics.snapshot().frames_dropped;
                        if drops != last_drops {
                            segmenter.reset_state();
                            if let Some(s) = &mut smoother {
                                s.reset();
                            }
                            last_drops = drops;
                        }
                        if !gate.should_infer(&frame, Instant::now()) {
                            continue; // scene idle: previous mask still fits
                        }
                        match segmenter.segment(&frame) {
                            Ok(mask) => {
                                let mask = match &mut smoother {
                                    Some(s) => s.update(&mask),
                                    None => mask,
                                };
                                shared.mask_slot.store(Arc::new(mask));
                                shared.metrics.on_mask_computed();
                            }
                            Err(e) => {
                                shared.metrics.on_mask_error();
                                tracing::warn!("segmentation error (keeping previous mask): {e}");
                            }
                        }
                    }
                    Err(crossbeam_channel::RecvTimeoutError::Timeout) => continue,
                    Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
                }
            }
        })
        .expect("spawn inference thread")
}

/// Spawns the compose worker: drains background-mode updates, blends the
/// latest frame with the dynamic mask, fans out to the preview broadcast,
/// writes to the sink, and paces to `max_fps`.
pub(crate) fn spawn_compose(
    shared: Arc<PipelineShared>,
    compositor: Box<dyn Compositor>,
    sink: Box<dyn FrameSink>,
    input_rx: Receiver<FrameBuffer>,
) -> JoinHandle<()> {
    std::thread::Builder::new()
        .name("frameiru-compose".into())
        .spawn(move || {
            let mut compositor = compositor;
            let mut sink = sink;
            let mut output: Option<FrameBuffer> = None;
            let mut pacer = Pacer::new(shared.config.max_fps);
            // Idle-composite gating state (U9.6): when the frame is unchanged
            // and neither the mask nor the background moved, reuse the last
            // composite instead of re-blurring/re-blending.
            let mut last_sig: Vec<u8> = Vec::new();
            let mut has_sig = false;
            let mut last_mask: Option<Arc<Mask>> = None;
            let mut last_bg: Option<BackgroundMode> = None;
            let mut bg_dirty = true;
            while !shared.stop.load(Ordering::Relaxed) {
                let frame = match input_rx.recv_timeout(STOP_POLL) {
                    Ok(frame) => frame,
                    Err(crossbeam_channel::RecvTimeoutError::Timeout) => continue,
                    Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
                };

                // Apply any pending background changes before this frame.
                while let Ok(mode) = shared.mode_rx.try_recv() {
                    if let Err(e) = compositor.update_background(mode.clone()) {
                        tracing::warn!("background update rejected ({mode:?}): {e}");
                    } else {
                        bg_dirty = true;
                        shared.background.store(Arc::new(mode));
                    }
                }

                let out = output.get_or_insert_with(|| FrameBuffer::new(frame.metadata));
                // Without a segmenter (or before its first mask) the slot is
                // empty-invalid or sized for an old resolution: substitute
                // an all-foreground mask so composition still works.
                let mask = shared.mask_slot.load_full();
                let mask = if mask.resolution != frame.metadata.resolution
                    || mask.data.len() < frame.metadata.resolution.area() as usize
                {
                    let filled = Arc::new(Mask::filled(frame.metadata.resolution, 1.0));
                    shared.mask_slot.store(filled);
                    shared.mask_slot.load_full()
                } else {
                    mask
                };

                // Scene signature for idle detection (cheap; failing to
                // produce one just disables the skip).
                let sig = crate::motion::luma_signature(&frame).ok();
                let bg = shared.background.load();
                let idle = !bg_dirty
                    && has_sig
                    && last_mask.is_some()
                    && matches!(
                        &**bg,
                        BackgroundMode::Blur { .. } | BackgroundMode::Image { .. }
                    )
                    && Arc::ptr_eq(&mask, last_mask.as_ref().expect("checked"))
                    && last_bg.as_ref() == Some(&**bg)
                    && sig.as_ref().is_some_and(|s| {
                        crate::motion::luma_diff(s, &last_sig)
                            < shared.config.compose_idle_threshold
                    });
                if idle {
                    shared.metrics.on_composite_skip();
                } else {
                    if let Err(e) = compositor.composite(&frame, &mask, out) {
                        tracing::warn!("composite error: {e}");
                        continue;
                    }
                    last_mask = Some(mask);
                    last_bg = Some((**bg).clone());
                    bg_dirty = false;
                }
                if let Some(s) = sig {
                    last_sig = s;
                    has_sig = true;
                }

                let _ = shared.preview_tx.send(Arc::new(out.clone()));
                if let Err(e) = sink.write_frame(out) {
                    tracing::warn!("sink error: {e}");
                }
                shared
                    .last_resolution
                    .store(Arc::new(frame.metadata.resolution));
                shared.metrics.on_composite();
                shared.metrics.record_latency(
                    frameiru_core::timestamp_us_now(),
                    frame.metadata.timestamp_us,
                );
                pacer.wait();
            }
        })
        .expect("spawn compose thread")
}

/// How often workers poll the stop flag while idle on their channels.
const STOP_POLL: Duration = Duration::from_millis(50);

/// Frame-rate limiter for the compose loop; `max_fps == 0` disables pacing.
struct Pacer {
    interval: Duration,
    next: Instant,
}

impl Pacer {
    fn new(max_fps: u32) -> Self {
        Self {
            interval: if max_fps == 0 {
                Duration::ZERO
            } else {
                Duration::from_secs_f64(1.0 / max_fps as f64)
            },
            next: Instant::now(),
        }
    }

    fn wait(&mut self) {
        if self.interval.is_zero() {
            return;
        }
        let now = Instant::now();
        if now < self.next {
            std::thread::sleep(self.next - now);
        }
        // If we fell behind, don't try to catch up in a burst; re-anchor.
        self.next = now.max(self.next) + self.interval;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pacer_uncapped_is_noop() {
        let mut p = Pacer::new(0);
        let t0 = Instant::now();
        p.wait();
        assert!(t0.elapsed() < Duration::from_millis(5));
    }

    #[test]
    fn pacer_enforces_min_interval() {
        // Coarse timing: 30ms slots, allow generous scheduler slack.
        let mut p = Pacer::new(33);
        p.wait(); // anchor the slot
        let t0 = Instant::now();
        p.wait(); // back-to-back: must wait out the remaining slot
        let elapsed = t0.elapsed();
        assert!(
            elapsed >= Duration::from_millis(20),
            "back-to-back waits must pace, slept {elapsed:?}"
        );
    }
}
