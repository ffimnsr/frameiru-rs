//! Slint app: wires the pipeline to the UI (preview, background controls,
//! device selection) and supports headless preview without a loopback
//! module (mock source + null sink).

use std::cell::RefCell;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::bail;
#[cfg(feature = "v4l2")]
use anyhow::Context as _;
use frameiru_core::format::Resolution;
use frameiru_core::traits::{FrameSink, FrameSource, Segmenter};
use frameiru_core::{BackgroundMode, FrameBuffer, OverlayMode};
use frameiru_pipeline::{Engine, PipelineConfig};
use frameiru_webcam_utils::controls::ControlInfo;
use slint::{CloseRequestResponse, ComponentHandle, SharedString, VecModel};

use crate::devices::{detect_devices, DeviceEntry};
// Re-exported so `main.rs` can construct the window.
pub(crate) use crate::gen::{FrameiruTray, MainWindow, MiniPreview};
use crate::mini_preview::MiniPreviewController;
use crate::virtual_camera::{self, SinkState};
use crate::webcam;

// Generated per-.slint-file modules are pulled in via `gen` (one module per
// compiled file, because `slint::include_modules!()` can only expand to a
// single generated file).

const PREVIEW_WIDTH: u32 = 640;
const PREVIEW_HEIGHT: u32 = 480;
const PREVIEW_RESOLUTION: Resolution = Resolution {
    width: PREVIEW_WIDTH,
    height: PREVIEW_HEIGHT,
};

/// Application state bridging the pipeline and the Slint window.
pub struct App {
    window: MainWindow,
    engine: Option<Engine>,
    devices: Vec<DeviceEntry>,
    model: Option<std::path::PathBuf>,
    model_input: Resolution,
    has_segmenter: bool,
    /// Loopback node the running engine publishes to (`None` = mock sink).
    sink_path: Option<PathBuf>,
    /// Last virtual-camera probe (displayed whenever the engine is idle).
    sink_state: SinkState,
    /// A privileged load/retry is in flight; the banner button is disabled.
    sink_action_busy: bool,
    /// Hot-attach watcher while the engine runs on a mock sink (layer 5).
    sink_watcher: Option<virtual_camera::SinkWatcher>,
    /// System tray icon; holding it keeps the app alive when the window is hidden.
    tray: FrameiruTray,
    /// Picture-in-picture mini preview, shown when the window is minimized.
    mini: MiniPreviewController,
}

impl App {
    pub fn new(
        window: MainWindow,
        model: Option<std::path::PathBuf>,
        model_input: Resolution,
    ) -> anyhow::Result<Self> {
        let devices = detect_devices();
        // Default to first physical camera if present, otherwise mock
        let default_index = if devices.len() > 1 { 1 } else { 0 };

        window.set_devices(slint::ModelRc::new(VecModel::from(
            devices
                .iter()
                .map(|d| SharedString::from(d.display_name.as_str()))
                .collect::<Vec<_>>(),
        )));
        window.set_selected_device(default_index);
        window.set_is_running(false);
        window.set_error_text("".into());

        let model_label = match &model {
            Some(path) => path
                .file_name()
                .map(|f| f.to_string_lossy().into_owned())
                .unwrap_or_else(|| path.display().to_string()),
            None => {
                #[cfg(feature = "onnx")]
                {
                    "Fusion: MediaPipe + RVM (embedded)".to_string()
                }
                #[cfg(not(feature = "onnx"))]
                {
                    "none (onnx feature disabled)".to_string()
                }
            }
        };
        window.set_model_name(SharedString::from(model_label));

        // Show the webcam control drawer only for a supported Anker C200.
        let c200_selected = devices
            .get(default_index as usize)
            .is_some_and(|d| d.is_anker_c200);
        window.set_webcam_available(c200_selected);
        if c200_selected {
            webcam::refresh_webcam_state(&window, &devices);
        }

        // Virtual camera status: honest banner + pill from the start.
        let sink_state = SinkState::probe();
        virtual_camera::apply(&window, &virtual_camera::view_for(&sink_state, ""));

        // PiP controller borrows a weak window handle before the window is
        // moved into the struct below.
        let mini = MiniPreviewController::new(window.as_weak())?;

        Ok(Self {
            window,
            engine: None,
            devices,
            model,
            model_input,
            has_segmenter: false,
            sink_path: None,
            sink_state,
            sink_action_busy: false,
            sink_watcher: None,
            tray: FrameiruTray::new()?,
            mini,
        })
    }

    /// Runs the event loop; returns after the window closes or the tray Quit
    /// action fires.
    pub fn run(self) -> anyhow::Result<()> {
        let app = Rc::new(RefCell::new(self));
        Self::wire_callbacks(app.clone());
        Self::wire_tray_callbacks(app.clone());
        let window = app.borrow().window.clone_strong();
        // Close-to-tray: closing the window hides it instead of tearing down
        // the app; the visible tray icon keeps the event loop alive.
        window
            .window()
            .on_close_requested(|| CloseRequestResponse::HideWindow);
        // Tray must be visible before the loop runs, or closing the window
        // releases the last keepalive and the loop exits.
        app.borrow().tray.show()?;
        window.run()?;
        // Loop exited (window closed with no tray, or tray Quit): shut the
        // pipeline down cleanly.
        if let Some(engine) = app.borrow_mut().engine.take() {
            engine.shutdown();
        }
        Ok(())
    }

    fn wire_tray_callbacks(app: Rc<RefCell<App>>) {
        // Tray left-click / "Show Frameiru": bring the window back up.
        let a = app.clone();
        app.borrow().tray.on_show_window(move || {
            if let Some(window) = a.borrow().window.as_weak().upgrade() {
                let _ = window.show();
            }
        });

        // Tray "Quit": shut the pipeline down and exit the event loop.
        let a = app.clone();
        app.borrow().tray.on_quit(move || {
            if let Some(engine) = a.borrow_mut().engine.take() {
                engine.shutdown();
            }
            let _ = slint::quit_event_loop();
        });

        // Tray "Mini preview when minimized": flip the checked state (the
        // platform checkbox mirrors whatever muda does) and toggle the PiP.
        let a = app.clone();
        app.borrow().tray.on_toggle_mini_preview(move || {
            let app = a.borrow();
            let next = !app.tray.get_mini_preview_checked();
            app.tray.set_mini_preview_checked(next);
            app.mini.set_enabled(next);
        });
    }

    fn wire_callbacks(app: Rc<RefCell<App>>) {
        let a = app.clone();
        app.borrow()
            .window
            .on_start_pipeline(move || Self::start_pipeline(&a));

        let a = app.clone();
        app.borrow().window.on_stop_pipeline(move || {
            let mut app = a.borrow_mut();
            Self::stop_engine(&mut app);
        });

        // Virtual camera banner action: LOAD / RELOAD (pkexec helper) or
        // RETRY (fresh probe). Runs off-thread so the UI stays live.
        let a = app.clone();
        app.borrow().window.on_enable_virtual_camera(move || {
            Self::request_virtual_camera(&a);
        });

        let a = app.clone();
        app.borrow().window.on_device_selected(move |_device| {
            let app = a.borrow();
            app.window.set_error_text("".into());
            let c200_selected = app.selected_webcam().is_some();
            app.window.set_webcam_available(c200_selected);
            if c200_selected {
                webcam::refresh_webcam_state(&app.window, &app.devices);
                app.window.set_webcam_drawer_open(true);
            } else {
                app.window.set_webcam_drawer_open(false);
            }
        });

        // Webcam drawer controls (Anker C200 only; callbacks are inert otherwise).
        let a = app.clone();
        app.borrow().window.on_webcam_set_fov(move |value| {
            Self::set_webcam_control(&a, webcam::webcam_control("fov"), value.as_str());
        });

        let a = app.clone();
        app.borrow().window.on_webcam_set_hdr(move |value| {
            Self::set_webcam_control(
                &a,
                webcam::webcam_control("hdr"),
                if value { "on" } else { "off" },
            );
        });

        let a = app.clone();
        app.borrow().window.on_webcam_set_flip(move |value| {
            Self::set_webcam_control(
                &a,
                webcam::webcam_control("horizontal_flip"),
                if value { "on" } else { "off" },
            );
        });

        let a = app.clone();
        app.borrow().window.on_webcam_set_vertical(move |value| {
            Self::set_webcam_control(
                &a,
                webcam::webcam_control("vertical_screen"),
                if value { "on" } else { "off" },
            );
        });

        let a = app.clone();
        app.borrow()
            .window
            .on_webcam_set_anti_flicker(move |value| {
                Self::set_webcam_control(
                    &a,
                    webcam::webcam_control("anti_flicker"),
                    if value { "1" } else { "0" },
                );
            });

        let a = app.clone();
        app.borrow().window.on_webcam_set_brightness(move |value| {
            Self::set_webcam_control(&a, webcam::webcam_control("brightness"), &value.to_string());
        });

        let a = app.clone();
        app.borrow().window.on_set_passthrough(move || {
            Self::set_background(&a, BackgroundMode::Passthrough);
        });

        let a = app.clone();
        app.borrow().window.on_set_blur(move |radius| {
            Self::set_background(&a, BackgroundMode::Blur { radius });
        });

        let a = app.clone();
        app.borrow().window.on_set_color(move |hex| {
            if let Ok((r, g, b)) = parse_hex_color(hex.as_str()) {
                Self::set_background(&a, BackgroundMode::Color { r, g, b });
            }
        });

        let a = app.clone();
        app.borrow().window.on_pick_image(move || {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("images", &["png", "jpg", "jpeg"])
                .pick_file()
            {
                Self::set_background(&a, BackgroundMode::Image { path });
            }
        });

        let a = app.clone();
        app.borrow().window.on_pick_video(move || {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("videos", &["mp4", "webm", "mov", "mkv"])
                .pick_file()
            {
                Self::set_background(&a, BackgroundMode::Video { path });
            }
        });

        let a = app.clone();
        app.borrow().window.on_set_overlay(move |name| {
            if let Some(mode) = parse_overlay(name.as_str()) {
                Self::set_overlay(&a, mode);
            }
        });
    }

    fn set_background(app: &Rc<RefCell<App>>, mode: BackgroundMode) {
        let app = app.borrow();
        if let Some(engine) = &app.engine {
            if let Err(e) = engine.handle().update_background(mode) {
                eprintln!("background update rejected: {e}");
            }
        }
    }

    fn set_overlay(app: &Rc<RefCell<App>>, mode: OverlayMode) {
        let app = app.borrow();
        if let Some(engine) = &app.engine {
            if let Err(e) = engine.handle().update_overlay(mode) {
                eprintln!("overlay update rejected: {e}");
            }
        }
    }

    /// Stops the running engine and resets the runtime UI state; the sink
    /// view reverts to the last probe. Idempotent.
    fn stop_engine(app: &mut App) {
        if let Some(engine) = app.engine.take() {
            engine.shutdown();
        }
        app.has_segmenter = false;
        app.sink_path = None;
        app.sink_watcher = None;
        app.window.set_is_running(false);
        app.window.set_fps_text("0.0".into());
        app.window.set_latency_text("0.0ms".into());
        app.window.set_error_text("".into());
        app.window.set_sink_runtime_error("".into());
        virtual_camera::apply(&app.window, &virtual_camera::view_for(&app.sink_state, ""));
    }

    /// Banner action handler: LOAD/RELOAD via the pkexec helper, or a fresh
    /// probe for RETRY states. Runs the helper off the UI thread. With the
    /// hot-attach watcher (layer 5) an engine that stayed running simply
    /// picks up the new device without a restart; only a stopped engine that
    /// we stopped for a module reload gets restarted.
    fn request_virtual_camera(app: &Rc<RefCell<App>>) {
        let restart_needed = {
            let mut a = app.borrow_mut();
            if a.sink_action_busy {
                return;
            }
            a.sink_action_busy = true;
            a.window.set_sink_action_busy(true);
            // A module reload needs the module free, so release our own
            // writer first; that engine must then be restarted.
            let restart_needed = a.engine.is_some() && a.sink_path.is_some();
            if restart_needed {
                Self::stop_engine(&mut a);
            }
            restart_needed
        };

        // Fresh probe while nothing of ours holds the device.
        let state = {
            let a = app.borrow();
            virtual_camera::refresh(&a.window)
        };

        if !state.needs_helper() {
            // Retry-only states: nothing privileged to do, probe is applied.
            // A running mock engine self-heals via the watcher; an idle
            // engine picks the device up on its next START.
            let mut a = app.borrow_mut();
            a.sink_action_busy = false;
            a.window.set_sink_action_busy(false);
            return;
        }

        let helper = virtual_camera::find_helper();
        if helper.is_none() {
            // No privileged helper anywhere: show the manual fallback and
            // bail out immediately.
            let fail_msg = format!("LOAD FAILED — {}", virtual_camera::run_helper(None).output);
            {
                let mut a = app.borrow_mut();
                a.sink_action_busy = false;
                a.window.set_sink_action_busy(false);
                a.window.set_sink_message(fail_msg.clone().into());
            }
            if restart_needed {
                Self::start_pipeline(app);
                // start_pipeline re-applies the state view; keep the detail.
                app.borrow()
                    .window
                    .set_sink_message(fail_msg.clone().into());
            }
            return;
        }
        app.borrow()
            .window
            .set_sink_message("Waiting for authorization…".into());

        let slot: Arc<Mutex<Option<virtual_camera::HelperOutcome>>> = Arc::new(Mutex::new(None));
        {
            let slot = Arc::clone(&slot);
            std::thread::spawn(move || {
                let outcome = virtual_camera::run_helper(helper.as_deref());
                *slot.lock().expect("helper slot mutex poisoned") = Some(outcome);
            });
        }
        Self::poll_helper_slot(app.clone(), slot, restart_needed);
    }

    /// Polls the helper thread's outcome on the UI thread (a `slint::Timer`
    /// may capture `Rc<RefCell<App>>` because it never leaves this thread),
    /// then re-probes, re-applies the banner, and restarts the engine when it
    /// was stopped for a module reload.
    fn poll_helper_slot(
        app: Rc<RefCell<App>>,
        slot: Arc<Mutex<Option<virtual_camera::HelperOutcome>>>,
        restart_needed: bool,
    ) {
        const POLL_EVERY: Duration = Duration::from_millis(150);
        const MAX_ATTEMPTS: u32 = 120; // ~18s for the auth dialog

        fn poll_once(
            app: &Rc<RefCell<App>>,
            slot: Arc<Mutex<Option<virtual_camera::HelperOutcome>>>,
            restart_needed: bool,
            attempts_left: u32,
        ) {
            let done = slot.lock().expect("helper slot mutex poisoned").take();
            match done {
                Some(outcome) => {
                    let failure_output = (!outcome.ok).then(|| outcome.output.clone());
                    let fail_msg = failure_output
                        .as_ref()
                        .map(|output| format!("LOAD FAILED — {output}."));
                    let mut a = app.borrow_mut();
                    a.sink_action_busy = false;
                    a.window.set_sink_action_busy(false);
                    if outcome.ok {
                        // No full probe here: the watcher may have hot-attached
                        // a real sink during the auth dialog, and probing our
                        // own device would report it as busy. Show what the
                        // engine actually publishes to.
                        let state = App::applied_sink_state(&a);
                        a.sink_state = state.clone();
                        virtual_camera::apply(&a.window, &virtual_camera::view_for(&state, ""));
                    } else if let Some(msg) = &fail_msg {
                        a.window.set_sink_message(msg.clone().into());
                    }
                    drop(a);
                    if restart_needed {
                        App::start_pipeline(app);
                        // start_pipeline re-applies the state view, which
                        // would hide the failure; re-surface the detail.
                        if let Some(msg) = &fail_msg {
                            app.borrow().window.set_sink_message(msg.clone().into());
                        }
                    } else if outcome.ok {
                        // Engine kept running on a mock sink: the watcher will
                        // hot-attach; nudge it now instead of waiting a tick.
                        App::sink_watch_tick(app);
                    }
                }
                None if attempts_left > 1 => {
                    let slot = Arc::clone(&slot);
                    let app = app.clone();
                    slint::Timer::single_shot(POLL_EVERY, move || {
                        poll_once(&app, slot, restart_needed, attempts_left - 1);
                    });
                }
                None => {
                    // Give up (dialog left open past the budget): restore the
                    // engine we stopped for a reload, stay quiet otherwise.
                    // Refresh-without-probe for the same reason as above: the
                    // engine may already publish via the watcher.
                    let mut a = app.borrow_mut();
                    a.sink_action_busy = false;
                    a.window.set_sink_action_busy(false);
                    let state = App::applied_sink_state(&a);
                    a.sink_state = state.clone();
                    virtual_camera::apply(&a.window, &virtual_camera::view_for(&state, ""));
                    drop(a);
                    if restart_needed {
                        App::start_pipeline(app);
                    }
                }
            }
        }

        poll_once(&app, slot, restart_needed, MAX_ATTEMPTS);
    }

    fn start_pipeline(app: &Rc<RefCell<App>>) {
        // The watcher (hot sink attach while the engine runs on a mock sink)
        // starts after the engine is up; keep the borrow short for that.
        let watch_needed = {
            let mut app = app.borrow_mut();
            if app.engine.is_some() {
                return;
            }
            app.window.set_error_text("".into());

            let source: Box<dyn FrameSource> = match build_source(&app.devices, &app.window) {
                Ok(source) => source,
                Err(e) => {
                    app.window.set_is_running(false);
                    app.window
                        .set_error_text(format!("Source error: {e}").into());
                    return;
                }
            };
            let actual_resolution = source.resolution();
            let (segmenter, has_segmenter) = match build_segmenter(&app.model, app.model_input) {
                Ok(Some(s)) => (Some(s), true),
                Ok(None) => (None, false),
                Err(e) => {
                    app.window.set_is_running(false);
                    app.window
                        .set_error_text(format!("Model error: {e}").into());
                    return;
                }
            };
            app.has_segmenter = has_segmenter;

            // Fresh probe: nothing of ours holds the loopback device yet, so
            // the open test reflects the real world (another app, perms, ...).
            app.sink_state = SinkState::probe();
            let (sink, sink_path) = build_sink(actual_resolution);
            app.sink_path = sink_path;
            let compositor = frameiru_compose::new_compositor();

            match Engine::start(
                PipelineConfig {
                    max_fps: 30,
                    infer_max_fps: 30,
                    ..Default::default()
                },
                source,
                segmenter,
                compositor,
                sink,
            ) {
                Ok(engine) => {
                    let rx = engine.handle().subscribe();
                    let handle = engine.handle();
                    // Every start defaults the effect to passthrough.
                    if let Err(e) = handle.update_background(BackgroundMode::Passthrough) {
                        eprintln!("passthrough default rejected: {e}");
                    }
                    app.window.set_active_effect("passthrough".into());
                    // Re-apply the selected overlay (the engine starts with none).
                    if let Some(overlay) = parse_overlay(&app.window.get_overlay_effect()) {
                        handle.update_overlay(overlay).unwrap_or_else(|e| {
                            eprintln!("overlay apply after start rejected: {e}");
                        });
                    }
                    app.window.set_is_running(true);
                    app.window.set_error_text("".into());
                    app.window.set_sink_runtime_error("".into());
                    // Reflect the sink the engine actually publishes to.
                    let state = Self::applied_sink_state(&app);
                    app.sink_state = state.clone();
                    virtual_camera::apply(&app.window, &virtual_camera::view_for(&state, ""));
                    Self::pump_preview(app.window.as_weak(), app.mini.weak(), rx, handle);
                    app.engine = Some(engine);
                    // Watch for a loopback device only when publishing to
                    // nothing yet (mock sink) — a real sink is already live.
                    app.sink_path.is_none()
                }
                Err(e) => {
                    app.sink_path = None;
                    app.window.set_is_running(false);
                    app.window
                        .set_error_text(format!("Start failed: {e}").into());
                    false
                }
            }
        };
        if watch_needed {
            Self::start_sink_watch(app);
        }
    }

    /// Starts the hot-attach watcher (1s poll) while the engine runs on a
    /// mock sink; the moment a usable loopback device appears it is swapped
    /// in without restarting the engine.
    fn start_sink_watch(app: &Rc<RefCell<App>>) {
        let mut a = app.borrow_mut();
        if a.engine.is_none() || a.sink_path.is_some() || a.sink_watcher.is_some() {
            return;
        }
        let watcher = virtual_camera::SinkWatcher::new();
        let ticker = app.clone();
        watcher.start(Duration::from_secs(1), move || {
            Self::sink_watch_tick(&ticker)
        });
        a.sink_watcher = Some(watcher);
    }

    /// The state the UI should show given what the engine actually publishes
    /// to: a live loopback node reads as Ready (probing would misreport our
    /// own device as busy), otherwise the last probe stands.
    fn applied_sink_state(app: &App) -> SinkState {
        match &app.sink_path {
            Some(path) => SinkState::Ready {
                exclusive: loopback_exclusive_caps(path),
                path: path.clone(),
            },
            None => app.sink_state.clone(),
        }
    }

    /// One watcher tick: attach a real sink when possible, and reflect any
    /// state change (Attached or failing probe) in the banner/pill.
    fn sink_watch_tick(app: &Rc<RefCell<App>>) {
        let event = {
            let mut a = app.borrow_mut();
            let handle = a.engine.as_ref().map(|e| e.handle());
            let Some(handle) = handle else {
                return;
            };
            let Some(watcher) = a.sink_watcher.as_mut() else {
                return;
            };
            let resolution = handle.last_resolution().unwrap_or(PREVIEW_RESOLUTION);
            watcher.tick(&handle, resolution)
        };
        let Some(event) = event else {
            return;
        };

        let mut a = app.borrow_mut();
        match event {
            virtual_camera::WatchEvent::Attached { path } => {
                // Live real sink: stop watching and show the honest Ready state.
                a.sink_watcher = None;
                a.sink_path = Some(path.clone());
                a.sink_state = SinkState::Ready {
                    exclusive: loopback_exclusive_caps(&path),
                    path,
                };
                virtual_camera::apply(&a.window, &virtual_camera::view_for(&a.sink_state, ""));
            }
            virtual_camera::WatchEvent::Probe(state) => {
                a.sink_state = state.clone();
                virtual_camera::apply(&a.window, &virtual_camera::view_for(&state, ""));
            }
        }
    }

    /// Spawns the preview pump: converts broadcast frames to slint images
    /// on the UI thread and posts them cleanly.
    fn pump_preview(
        window: slint::Weak<MainWindow>,
        mini: slint::Weak<MiniPreview>,
        mut rx: tokio::sync::broadcast::Receiver<Arc<FrameBuffer>>,
        handle: frameiru_pipeline::PipelineHandle,
    ) {
        let is_rendering = Arc::new(AtomicBool::new(false));
        let frame_counter = Arc::new(std::sync::atomic::AtomicU64::new(0));
        std::thread::Builder::new()
            .name("frameiru-ui-preview".into())
            .spawn(move || loop {
                match rx.blocking_recv() {
                    Ok(frame) => {
                        let count = frame_counter.fetch_add(1, Ordering::Relaxed);
                        // Skip if the previous frame is still waiting or rendering
                        if is_rendering.load(Ordering::Acquire) {
                            continue;
                        }

                        let update_telemetry = count.is_multiple_of(15);
                        let telemetry = if update_telemetry {
                            let m = handle.metrics();
                            Some((
                                format!("{:.1}", m.composite_fps),
                                format!("{:.1}ms", m.latency_us as f64 / 1000.0),
                                handle.sink_error(),
                            ))
                        } else {
                            None
                        };

                        is_rendering.store(true, Ordering::Release);
                        let weak = window.clone();
                        let mini = mini.clone();
                        let is_rendering_flag = Arc::clone(&is_rendering);
                        let invoke_res = slint::invoke_from_event_loop(move || {
                            if let Some(win) = weak.upgrade() {
                                if let Some(image) =
                                    frameiru_core::slint_compat::frame_to_slint_image(&frame)
                                {
                                    win.set_preview(image.clone());
                                    // Same shared pixel buffer, zero extra
                                    // copies; PiP window re-renders it.
                                    if let Some(mini) = mini.upgrade() {
                                        mini.set_preview(image);
                                    }
                                }
                                if let Some((fps, lat, sink_err)) = telemetry {
                                    win.set_fps_text(fps.into());
                                    win.set_latency_text(lat.into());
                                    // Sink write failures surface as a red
                                    // SINK: ERROR pill + banner (cleared on
                                    // recovery by the pipeline).
                                    win.set_sink_runtime_error(sink_err.unwrap_or_default().into());
                                }
                            }
                            is_rendering_flag.store(false, Ordering::Release);
                        });

                        if invoke_res.is_err() {
                            is_rendering.store(false, Ordering::Release);
                            break;
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                }
            })
            .expect("spawn preview pump");
    }

    /// The selected device when it is a supported webcam (Anker C200).
    fn selected_webcam(&self) -> Option<&DeviceEntry> {
        let index = self.window.get_selected_device() as usize;
        self.devices.get(index).filter(|d| d.is_anker_c200)
    }

    /// Opens the selected webcam, writes a control, then reflects the
    /// readback value in the drawer (truth-in-UI even when the set fails).
    fn set_webcam_control(app: &Rc<RefCell<App>>, info: &'static ControlInfo, value: &str) {
        let app = app.borrow();
        let Some(device) = app.selected_webcam() else {
            return;
        };
        let fd = match File::options().read(true).write(true).open(&device.id) {
            Ok(fd) => fd,
            Err(e) => {
                app.window
                    .set_webcam_error(format!("cannot open {}: {e}", device.id).into());
                return;
            }
        };
        if let Err(e) = webcam::write_webcam_value(&fd, info, value) {
            app.window
                .set_webcam_error(format!("set {} failed: {e:#}", info.name).into());
        }
        webcam::push_webcam_control(&app.window, info, &fd);
    }
}

fn build_source(
    devices: &[DeviceEntry],
    window: &MainWindow,
) -> anyhow::Result<Box<dyn FrameSource>> {
    let index = window.get_selected_device() as usize;
    let Some(selected) = devices.get(index) else {
        bail!("no device selected");
    };
    let resolution = Resolution {
        width: PREVIEW_WIDTH,
        height: PREVIEW_HEIGHT,
    };
    if selected.id == "(mock)" {
        return Ok(Box::new(frameiru_capture::MockSource::new(resolution)?));
    }
    #[cfg(feature = "v4l2")]
    {
        let source = frameiru_capture::v4l2::V4l2Source::open_with_resolution(
            &selected.id,
            Some(resolution),
        )
        .with_context(|| format!("cannot open {}", selected.display_name))?;
        Ok(Box::new(source))
    }
    #[cfg(not(feature = "v4l2"))]
    bail!("real cameras need the v4l2 feature; only the mock source is available")
}

fn build_segmenter(
    model: &Option<std::path::PathBuf>,
    input_size: Resolution,
) -> anyhow::Result<Option<Box<dyn Segmenter>>> {
    #[cfg(feature = "onnx")]
    {
        use frameiru_segment::{load_embedded, load_model, OnnxConfig, EMBEDDED_MODEL_INPUT};
        let segmenter = match model {
            Some(path) => {
                let config = OnnxConfig::new(input_size)?;
                load_model(path, config)
                    .with_context(|| format!("cannot load model {}", path.display()))?
            }
            // No model picked: fall back to the embedded fusion model.
            None => {
                let mut config = OnnxConfig::new(EMBEDDED_MODEL_INPUT)?;
                config.normalization = frameiru_segment::embedded_normalization();
                load_embedded(config)?
            }
        };
        Ok(Some(segmenter))
    }
    #[cfg(not(feature = "onnx"))]
    {
        let _ = input_size;
        if model.is_some() {
            bail!("segmentation needs the onnx feature (rebuild with --features onnx)")
        }
        Ok(None)
    }
}

/// Headless preview: prefer a loopback device matching the actual source
/// resolution, fall back to a null sink. Returns the opened node path so the
/// UI can report exactly where frames are published.
fn build_sink(source_resolution: Resolution) -> (Box<dyn FrameSink>, Option<PathBuf>) {
    #[cfg(feature = "v4l2")]
    {
        if let Some((sink, path)) = frameiru_sink::loopback::open_first_available(source_resolution)
        {
            return (Box::new(sink), Some(path));
        }
        eprintln!("no loopback device; running headless preview only");
    }
    let _ = source_resolution;
    (Box::new(frameiru_sink::MockSink::new()), None)
}

/// Per-device `exclusive_caps` for the running-sink badge. Unreadable caps
/// fall back to true (no nagging); builds without `v4l2` never publish, so
/// the flag is irrelevant there.
fn loopback_exclusive_caps(path: &Path) -> bool {
    #[cfg(feature = "v4l2")]
    {
        frameiru_sink::loopback::path_exclusive_caps(path).unwrap_or(true)
    }
    #[cfg(not(feature = "v4l2"))]
    {
        let _ = path;
        true
    }
}

/// Parses `#rrggbb` into an RGB tuple.
pub fn parse_hex_color(hex: &str) -> Result<(u8, u8, u8), String> {
    let hex = hex.strip_prefix('#').unwrap_or(hex);
    if hex.len() != 6 {
        return Err(format!("expected #rrggbb, got {hex:?}"));
    }
    let mut channels = [0u8; 3];
    for (i, chunk) in hex.as_bytes().chunks(2).enumerate() {
        let s = std::str::from_utf8(chunk).map_err(|_| "non-ascii color".to_string())?;
        channels[i] = u8::from_str_radix(s, 16).map_err(|_| format!("bad color {hex:?}"))?;
    }
    Ok((channels[0], channels[1], channels[2]))
}

/// Maps the overlay button name to its mode; unknown names yield `None`.
fn parse_overlay(name: &str) -> Option<OverlayMode> {
    match name {
        "none" => Some(OverlayMode::None),
        "scanlines" => Some(OverlayMode::Scanlines),
        "light_leak" => Some(OverlayMode::LightLeak),
        "crt" => Some(OverlayMode::Crt),
        "chromatic" => Some(OverlayMode::ChromaticAberration),
        "vhs" => Some(OverlayMode::Vhs),
        "dither" => Some(OverlayMode::Dither),
        "halftone" => Some(OverlayMode::Halftone),
        "grayscale" => Some(OverlayMode::Grayscale),
        "cyberpunk" => Some(OverlayMode::Cyberpunk),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_overlay_names() {
        assert_eq!(parse_overlay("none"), Some(OverlayMode::None));
        assert_eq!(parse_overlay("scanlines"), Some(OverlayMode::Scanlines));
        assert_eq!(parse_overlay("light_leak"), Some(OverlayMode::LightLeak));
        assert_eq!(parse_overlay("crt"), Some(OverlayMode::Crt));
        assert_eq!(
            parse_overlay("chromatic"),
            Some(OverlayMode::ChromaticAberration)
        );
        assert_eq!(parse_overlay("vhs"), Some(OverlayMode::Vhs));
        assert_eq!(parse_overlay("dither"), Some(OverlayMode::Dither));
        assert_eq!(parse_overlay("halftone"), Some(OverlayMode::Halftone));
        assert_eq!(parse_overlay("grayscale"), Some(OverlayMode::Grayscale));
        assert_eq!(parse_overlay("cyberpunk"), Some(OverlayMode::Cyberpunk));
        assert_eq!(parse_overlay("banana"), None);
        assert_eq!(parse_overlay(""), None);
    }

    #[test]
    fn parses_hex_colors() {
        assert_eq!(parse_hex_color("#ff0000").unwrap(), (255, 0, 0));
        assert_eq!(parse_hex_color("00ff00").unwrap(), (0, 255, 0));
        assert_eq!(parse_hex_color("#1e88e5").unwrap(), (30, 136, 229));
        for bad in ["#fff", "#gggggg", "#ff000", ""] {
            assert!(parse_hex_color(bad).is_err(), "{bad:?} must fail");
        }
    }
}
