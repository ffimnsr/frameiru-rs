//! Slint app: wires the pipeline to the UI (preview, background controls,
//! device selection) and supports headless preview without a loopback
//! module (mock source + null sink).

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{bail, Context as _};
use frameiru_core::format::Resolution;
use frameiru_core::traits::{FrameSink, FrameSource, Segmenter};
use frameiru_core::{BackgroundMode, FrameBuffer};
use frameiru_pipeline::{Engine, PipelineConfig};
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
            real_devices.push((path.to_string_lossy().to_string(), display_name));
        }

        real_devices.sort_by(|a, b| {
            let num_a = extract_trailing_number(&a.0);
            let num_b = extract_trailing_number(&b.0);
            match (num_a, num_b) {
                (Some(na), Some(nb)) => na.cmp(&nb),
                _ => a.0.cmp(&b.0),
            }
        });

        for (id, display_name) in real_devices {
            devices.push(DeviceEntry { id, display_name });
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
            a.borrow().window.set_error_text("".into());
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
    }

    fn set_background(app: &Rc<RefCell<App>>, mode: BackgroundMode) {
        let app = app.borrow();
        if let Some(engine) = &app.engine {
            if let Err(e) = engine.handle().update_background(mode) {
                eprintln!("background update rejected: {e}");
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

#[cfg(test)]
mod tests {
    use super::*;

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
