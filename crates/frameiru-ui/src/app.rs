//! Slint app: wires the pipeline to the UI (preview, background controls,
//! device selection) and supports headless preview without a loopback
//! module (mock source + null sink).

use std::cell::RefCell;
use std::fs::File;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{bail, Context as _};
use frameiru_core::format::Resolution;
use frameiru_core::traits::{FrameSink, FrameSource, Segmenter};
use frameiru_core::{BackgroundMode, FrameBuffer, OverlayMode};
use frameiru_pipeline::{Engine, PipelineConfig};
use frameiru_webcam_utils::controls::{
    find_control, format_bool, parse_bool, ControlInfo, ControlKind,
};
use frameiru_webcam_utils::{detect, fov, v4l2, vendor, Error as WebcamError};
use slint::{ComponentHandle, SharedString, VecModel};

slint::include_modules!();

const PREVIEW_WIDTH: u32 = 640;
const PREVIEW_HEIGHT: u32 = 480;

/// Represents an available video input device (mock source or physical V4L2 camera).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceEntry {
    /// Device identifier: `(mock)` or `/dev/videoN`.
    pub id: String,
    /// Friendly user-visible label: e.g. "Anker PowerConf C200 (/dev/video0)".
    pub display_name: String,
    /// True for supported webcams whose vendor controls are exposed in the drawer.
    pub is_anker_c200: bool,
}

/// Cleans raw V4L2 card/node name to remove truncated duplicate model strings.
/// E.g. "Anker PowerConf C200: Anker Pow" -> "Anker PowerConf C200".
pub fn clean_device_name(name: &str) -> String {
    let trimmed = name.trim();
    if let Some((head, tail)) = trimmed.split_once(':') {
        let h = head.trim();
        let t = tail.trim();
        if !h.is_empty() && (h.starts_with(t) || t.starts_with(h)) {
            return if h.len() >= t.len() {
                h.to_string()
            } else {
                t.to_string()
            };
        }
    }
    trimmed.to_string()
}

/// Extracts trailing digits from a device path for natural numeric sorting.
pub fn extract_trailing_number(path: &str) -> Option<u32> {
    path.rsplit(|c: char| !c.is_ascii_digit())
        .next()
        .and_then(|digits| digits.parse::<u32>().ok())
}

/// Enumerate available capture devices: filters out metadata-only and loopback devices.
pub fn detect_devices() -> Vec<DeviceEntry> {
    let mut devices = vec![DeviceEntry {
        id: "(mock)".into(),
        display_name: "Mock Camera ((mock))".into(),
        is_anker_c200: false,
    }];

    #[cfg(feature = "v4l2")]
    {
        let mut real_devices = Vec::new();
        for node in frameiru_capture::v4l::context::enum_devices() {
            let path = node.path();
            // Verify device can be opened and actually supports VIDEO_CAPTURE
            let Ok(dev) = frameiru_capture::v4l::Device::with_path(path) else {
                continue;
            };
            let Ok(caps) = dev.query_caps() else {
                continue;
            };
            if !caps
                .capabilities
                .contains(frameiru_capture::v4l::capability::Flags::VIDEO_CAPTURE)
            {
                continue;
            }
            // Skip loopback devices (e.g. Frameiru virtual camera sink)
            if caps.driver.starts_with("v4l2 loopback") || caps.card.contains("Frameiru") {
                continue;
            }

            let raw_name = if !caps.card.trim().is_empty() {
                caps.card
            } else if let Some(name) = node.name() {
                name
            } else {
                "Camera".to_string()
            };

            let clean_name = clean_device_name(&raw_name);
            let display_name = format!("{clean_name} ({})", path.display());
            let is_anker_c200 = detect::is_anker_c200(path);
            real_devices.push((
                path.to_string_lossy().to_string(),
                display_name,
                is_anker_c200,
            ));
        }

        real_devices.sort_by(|a, b| {
            let num_a = extract_trailing_number(&a.0);
            let num_b = extract_trailing_number(&b.0);
            match (num_a, num_b) {
                (Some(na), Some(nb)) => na.cmp(&nb),
                _ => a.0.cmp(&b.0),
            }
        });

        for (id, display_name, is_anker_c200) in real_devices {
            devices.push(DeviceEntry {
                id,
                display_name,
                is_anker_c200,
            });
        }
    }

    devices
}

/// Application state bridging the pipeline and the Slint window.
pub struct App {
    window: MainWindow,
    engine: Option<Engine>,
    devices: Vec<DeviceEntry>,
    model: Option<std::path::PathBuf>,
    model_input: Resolution,
    has_segmenter: bool,
}

impl App {
    pub fn new(
        window: MainWindow,
        model: Option<std::path::PathBuf>,
        model_input: Resolution,
    ) -> Self {
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
            refresh_webcam_state(&window, &devices);
        }

        Self {
            window,
            engine: None,
            devices,
            model,
            model_input,
            has_segmenter: false,
        }
    }

    /// Runs the event loop; returns after the window closes.
    pub fn run(self) -> anyhow::Result<()> {
        let app = Rc::new(RefCell::new(self));
        Self::wire_callbacks(app.clone());
        let window = app.borrow().window.clone_strong();
        window.run()?;
        // Window closed: shut the pipeline down cleanly.
        if let Some(engine) = app.borrow_mut().engine.take() {
            engine.shutdown();
        }
        Ok(())
    }

    fn wire_callbacks(app: Rc<RefCell<App>>) {
        let a = app.clone();
        app.borrow()
            .window
            .on_start_pipeline(move || Self::start_pipeline(&a));

        let a = app.clone();
        app.borrow().window.on_stop_pipeline(move || {
            let mut app = a.borrow_mut();
            if let Some(engine) = app.engine.take() {
                engine.shutdown();
            }
            app.has_segmenter = false;
            app.window.set_is_running(false);
            app.window.set_fps_text("0.0".into());
            app.window.set_latency_text("0.0ms".into());
            app.window.set_error_text("".into());
        });

        let a = app.clone();
        app.borrow().window.on_device_selected(move |_device| {
            let app = a.borrow();
            app.window.set_error_text("".into());
            let c200_selected = app.selected_webcam().is_some();
            app.window.set_webcam_available(c200_selected);
            if c200_selected {
                refresh_webcam_state(&app.window, &app.devices);
                app.window.set_webcam_drawer_open(true);
            } else {
                app.window.set_webcam_drawer_open(false);
            }
        });

        // Webcam drawer controls (Anker C200 only; callbacks are inert otherwise).
        let a = app.clone();
        app.borrow().window.on_webcam_set_fov(move |value| {
            Self::set_webcam_control(&a, webcam_control("fov"), value.as_str());
        });

        let a = app.clone();
        app.borrow().window.on_webcam_set_hdr(move |value| {
            Self::set_webcam_control(&a, webcam_control("hdr"), if value { "on" } else { "off" });
        });

        let a = app.clone();
        app.borrow().window.on_webcam_set_flip(move |value| {
            Self::set_webcam_control(
                &a,
                webcam_control("horizontal_flip"),
                if value { "on" } else { "off" },
            );
        });

        let a = app.clone();
        app.borrow().window.on_webcam_set_vertical(move |value| {
            Self::set_webcam_control(
                &a,
                webcam_control("vertical_screen"),
                if value { "on" } else { "off" },
            );
        });

        let a = app.clone();
        app.borrow()
            .window
            .on_webcam_set_anti_flicker(move |value| {
                Self::set_webcam_control(
                    &a,
                    webcam_control("anti_flicker"),
                    if value { "1" } else { "0" },
                );
            });

        let a = app.clone();
        app.borrow().window.on_webcam_set_brightness(move |value| {
            Self::set_webcam_control(&a, webcam_control("brightness"), &value.to_string());
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

    fn start_pipeline(app: &Rc<RefCell<App>>) {
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

        let sink: Box<dyn FrameSink> = build_sink(actual_resolution);
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
                Self::pump_preview(app.window.as_weak(), rx, handle);
                app.engine = Some(engine);
            }
            Err(e) => {
                app.window.set_is_running(false);
                app.window
                    .set_error_text(format!("Start failed: {e}").into());
            }
        }
    }

    /// Spawns the preview pump: converts broadcast frames to slint images
    /// on the UI thread and posts them cleanly.
    fn pump_preview(
        window: slint::Weak<MainWindow>,
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
                            ))
                        } else {
                            None
                        };

                        is_rendering.store(true, Ordering::Release);
                        let weak = window.clone();
                        let is_rendering_flag = Arc::clone(&is_rendering);
                        let invoke_res = slint::invoke_from_event_loop(move || {
                            if let Some(win) = weak.upgrade() {
                                if let Some(image) =
                                    frameiru_core::slint_compat::frame_to_slint_image(&frame)
                                {
                                    win.set_preview(image);
                                }
                                if let Some((fps, lat)) = telemetry {
                                    win.set_fps_text(fps.into());
                                    win.set_latency_text(lat.into());
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
        if let Err(e) = write_webcam_value(&fd, info, value) {
            app.window
                .set_webcam_error(format!("set {} failed: {e:#}", info.name).into());
        }
        push_webcam_control(&app.window, info, &fd);
    }
}

/// Returns the statically registered control of `name` (always present).
fn webcam_control(name: &str) -> &'static ControlInfo {
    find_control(name).expect("webcam control registry is static")
}

/// Reads the current value of one control as display text.
fn read_webcam_value(fd: &File, info: &ControlInfo) -> Result<String, WebcamError> {
    match info.kind {
        ControlKind::VendorBool => Ok(format_bool(vendor::get_bool(fd, info.id as u8)?).into()),
        ControlKind::VendorU8 => Ok(vendor::get_u8(fd, info.id as u8)?.to_string()),
        ControlKind::VendorFov => Ok(fov::describe_value(fov::get(fd)?)),
        ControlKind::V4l2Bool => Ok(format_bool(v4l2::get(fd, info.id)? != 0).into()),
        ControlKind::V4l2Int | ControlKind::V4l2Menu => Ok(v4l2::get(fd, info.id)?.to_string()),
    }
}

/// Writes one control from its display text (mirrors the CLI `set` behavior).
fn write_webcam_value(fd: &File, info: &ControlInfo, value: &str) -> Result<(), WebcamError> {
    let invalid = || WebcamError::InvalidValue {
        name: info.name.to_owned(),
        value: value.to_owned(),
    };
    match info.kind {
        ControlKind::VendorBool => {
            let value = parse_bool(value).ok_or_else(invalid)?;
            vendor::set_bool(fd, info.id as u8, value)?;
        }
        ControlKind::VendorU8 => {
            let value = value.parse::<u8>().map_err(|_| invalid())?;
            vendor::set_u8(fd, info.id as u8, value)?;
        }
        ControlKind::VendorFov => {
            let value = fov::parse_value(value).ok_or_else(invalid)?;
            fov::set(fd, value)?;
        }
        ControlKind::V4l2Bool => {
            let value = parse_bool(value).ok_or_else(invalid)?;
            v4l2::set(fd, info.id, if value { 1 } else { 0 })?;
        }
        ControlKind::V4l2Int | ControlKind::V4l2Menu => {
            let value = value.parse::<i32>().map_err(|_| invalid())?;
            v4l2::set(fd, info.id, value)?;
        }
    }
    Ok(())
}

/// Syncs the drawer widget for one control with its live device value.
/// Failed reads leave the current widget state untouched.
fn push_webcam_control(window: &MainWindow, info: &ControlInfo, fd: &File) {
    let Ok(text) = read_webcam_value(fd, info) else {
        return;
    };
    match info.name {
        "fov" => {
            let raw = fov::get(fd).unwrap_or(0);
            let preset = fov::FOV_PRESETS
                .iter()
                .find(|(value, _, _)| *value == raw)
                .map(|(_, name, _)| *name)
                .unwrap_or("custom");
            window.set_webcam_fov(preset.into());
        }
        "hdr" => window.set_webcam_hdr(parse_bool(&text) == Some(true)),
        "horizontal_flip" => window.set_webcam_flip(parse_bool(&text) == Some(true)),
        "vertical_screen" => window.set_webcam_vertical(parse_bool(&text) == Some(true)),
        "anti_flicker" => window.set_webcam_anti_flicker(text != "0"),
        "brightness" => {
            if let Ok(value) = text.parse::<i32>() {
                window.set_webcam_brightness(value);
            }
        }
        _ => {}
    }
}

/// Read every drawer control from the selected webcam and sync the widgets.
fn refresh_webcam_state(window: &MainWindow, devices: &[DeviceEntry]) {
    window.set_webcam_error("".into());
    let index = window.get_selected_device() as usize;
    let Some(device) = devices.get(index).filter(|d| d.is_anker_c200) else {
        return;
    };
    let fd = match File::options().read(true).write(true).open(&device.id) {
        Ok(fd) => fd,
        Err(e) => {
            window
                .set_webcam_error(format!("webcam control: cannot open {}: {e}", device.id).into());
            return;
        }
    };
    for name in [
        "fov",
        "hdr",
        "horizontal_flip",
        "vertical_screen",
        "anti_flicker",
        "brightness",
    ] {
        push_webcam_control(window, webcam_control(name), &fd);
    }
    let brightness = webcam_control("brightness");
    if let Ok((min, max)) = v4l2::range(&fd, brightness.id) {
        window.set_webcam_brightness_min(min);
        window.set_webcam_brightness_max(max);
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

/// Headless preview: prefer a loopback device matching the actual source resolution,
/// fall back to a null sink.
fn build_sink(source_resolution: Resolution) -> Box<dyn FrameSink> {
    #[cfg(feature = "v4l2")]
    {
        if let Ok(sink) = frameiru_sink::v4l2::LoopbackSink::open("/dev/video10", source_resolution)
        {
            return Box::new(sink);
        }
        eprintln!("no loopback device; running headless preview only");
    }
    let _ = source_resolution;
    Box::new(frameiru_sink::MockSink::new())
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

    #[test]
    fn cleans_device_names() {
        assert_eq!(
            clean_device_name("Anker PowerConf C200: Anker Pow"),
            "Anker PowerConf C200"
        );
        assert_eq!(
            clean_device_name("Integrated Camera: Integrated C"),
            "Integrated Camera"
        );
        assert_eq!(
            clean_device_name("Logitech Webcam C920"),
            "Logitech Webcam C920"
        );
    }

    #[test]
    fn extracts_trailing_number_correctly() {
        assert_eq!(extract_trailing_number("/dev/video0"), Some(0));
        assert_eq!(extract_trailing_number("/dev/video1"), Some(1));
        assert_eq!(extract_trailing_number("/dev/video10"), Some(10));
        assert_eq!(extract_trailing_number("(mock)"), None);
    }

    #[test]
    fn detects_devices_contains_mock_and_valid_capture_nodes() {
        let devices = detect_devices();
        assert!(!devices.is_empty());
        assert_eq!(devices[0].id, "(mock)");
        assert!(!devices[0].is_anker_c200, "mock source is never a C200");
        // If real camera is present in test environment, verify metadata/loopback devices are excluded
        for dev in &devices[1..] {
            assert!(dev.id.starts_with("/dev/video"));
            assert_ne!(
                dev.id, "/dev/video10",
                "loopback output device must be excluded"
            );
            assert!(
                dev.display_name.contains(&dev.id),
                "display name must have device path hint"
            );
        }
    }
}
