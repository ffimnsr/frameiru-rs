//! Slint app: wires the pipeline to the UI (preview, background controls,
//! device selection) and supports headless preview without a loopback
//! module (mock source + null sink).

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use anyhow::{bail, Context as _};
use frameiru_core::format::Resolution;
use frameiru_core::traits::{FrameSink, FrameSource, Segmenter};
use frameiru_core::{BackgroundMode, FrameBuffer};
use frameiru_pipeline::{Engine, PipelineConfig, PipelineHandle};
use slint::{ComponentHandle, SharedString, VecModel};

slint::include_modules!();

const PREVIEW_WIDTH: u32 = 640;
const PREVIEW_HEIGHT: u32 = 480;

/// Application state bridging the pipeline and the Slint window.
pub struct App {
    window: MainWindow,
    handle: Option<PipelineHandle>,
    devices: Vec<String>,
    model: Option<std::path::PathBuf>,
    model_input: Resolution,
}

impl App {
    pub fn new(
        window: MainWindow,
        model: Option<std::path::PathBuf>,
        model_input: Resolution,
    ) -> Self {
        let mut app = Self {
            window,
            handle: None,
            devices: vec!["(mock)".into()],
            model,
            model_input,
        };
        app.refresh_devices();
        app
    }

    /// Populates the device selector (mock source + physical cameras).
    fn refresh_devices(&mut self) {
        let mut devices = vec!["(mock)".into()];
        #[cfg(feature = "v4l2")]
        {
            for node in frameiru_capture::v4l::context::enum_devices() {
                devices.push(node.path().display().to_string());
            }
        }
        self.devices = devices;
        self.window.set_devices(slint::ModelRc::new(VecModel::from(
            self.devices
                .iter()
                .map(|d| SharedString::from(d.as_str()))
                .collect::<Vec<_>>(),
        )));
    }

    /// Runs the event loop; returns after the window closes.
    pub fn run(self) -> anyhow::Result<()> {
        let app = Rc::new(RefCell::new(self));
        Self::wire_callbacks(app.clone());
        let window = app.borrow().window.clone_strong();
        window.run()?;
        // Window closed: shut the pipeline down cleanly.
        if let Some(handle) = app.borrow_mut().handle.take() {
            handle.shutdown();
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
            let app = a.borrow();
            if let Some(handle) = &app.handle {
                handle.shutdown();
            }
            app.set_status("stopped");
        });

        let a = app.clone();
        app.borrow().window.on_device_selected(move |device| {
            let app = a.borrow();
            app.set_status(&format!("device: {device}"));
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
        match &app.handle {
            Some(handle) => match handle.update_background(mode.clone()) {
                Ok(()) => app.set_status(&format!("background: {mode:?}")),
                Err(e) => app.set_status(&format!("background rejected: {e}")),
            },
            None => app.set_status("start the pipeline first"),
        }
    }

    fn start_pipeline(app: &Rc<RefCell<App>>) {
        let mut app = app.borrow_mut();
        if app.handle.is_some() {
            app.set_status("already running");
            return;
        }

        let resolution = Resolution {
            width: PREVIEW_WIDTH,
            height: PREVIEW_HEIGHT,
        };

        let source: Box<dyn FrameSource> = match build_source(&app.devices, &app.window) {
            Ok(source) => source,
            Err(e) => {
                app.set_status(&format!("source: {e}"));
                return;
            }
        };
        let segmenter: Option<Box<dyn Segmenter>> =
            match build_segmenter(&app.model, app.model_input) {
                Ok(segmenter) => segmenter,
                Err(e) => {
                    app.set_status(&format!("model: {e}"));
                    return;
                }
            };
        let sink: Box<dyn FrameSink> = build_sink();
        let compositor = frameiru_compose::new_compositor();

        match Engine::start(
            PipelineConfig {
                max_fps: 30,
                ..Default::default()
            },
            source,
            segmenter,
            compositor,
            sink,
        ) {
            Ok(engine) => {
                let handle = engine.handle();
                let rx = handle.subscribe();
                app.handle = Some(handle);
                app.set_status(&format!(
                    "running ({}x{})",
                    resolution.width, resolution.height
                ));
                drop(engine); // engine lives through the handle
                Self::pump_preview(app.window.as_weak(), rx);
            }
            Err(e) => app.set_status(&format!("start failed: {e}")),
        }
    }

    /// Spawns the preview pump: converts broadcast frames to slint images
    /// and posts them to the UI thread.
    fn pump_preview(
        window: slint::Weak<MainWindow>,
        mut rx: tokio::sync::broadcast::Receiver<Arc<FrameBuffer>>,
    ) {
        std::thread::Builder::new()
            .name("frameiru-ui-preview".into())
            .spawn(move || loop {
                match rx.blocking_recv() {
                    Ok(frame) => {
                        // Convert on the UI thread: slint images are not Send.
                        let weak = window.clone();
                        let _ = slint::invoke_from_event_loop(move || {
                            if let Some(win) = weak.upgrade() {
                                if let Some(image) =
                                    frameiru_core::slint_compat::frame_to_slint_image(&frame)
                                {
                                    win.set_preview(image);
                                }
                            }
                        });
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                }
            })
            .expect("spawn preview pump");
    }

    fn set_status(&self, text: &str) {
        self.window.set_status_text(text.into());
    }
}

fn build_source(devices: &[String], window: &MainWindow) -> anyhow::Result<Box<dyn FrameSource>> {
    let index = window.get_selected_device() as usize;
    let Some(selected) = devices.get(index) else {
        bail!("no device selected");
    };
    let resolution = Resolution {
        width: PREVIEW_WIDTH,
        height: PREVIEW_HEIGHT,
    };
    if selected == "(mock)" {
        return Ok(Box::new(frameiru_capture::MockSource::new(resolution)?));
    }
    #[cfg(feature = "v4l2")]
    {
        let source =
            frameiru_capture::v4l2::V4l2Source::open_with_resolution(selected, Some(resolution))
                .with_context(|| format!("cannot open {selected}"))?;
        Ok(Box::new(source))
    }
    #[cfg(not(feature = "v4l2"))]
    bail!("real cameras need the v4l2 feature; only the mock source is available")
}

fn build_segmenter(
    model: &Option<std::path::PathBuf>,
    input_size: Resolution,
) -> anyhow::Result<Option<Box<dyn Segmenter>>> {
    let Some(path) = model else {
        return Ok(None);
    };
    #[cfg(feature = "onnx")]
    {
        let config = frameiru_segment::OnnxConfig::new(input_size)?;
        let segmenter = frameiru_segment::load_model(path, config)
            .with_context(|| format!("cannot load model {}", path.display()))?;
        Ok(Some(segmenter))
    }
    #[cfg(not(feature = "onnx"))]
    {
        let _ = (path, input_size);
        bail!("segmentation needs the onnx feature (rebuild with --features onnx)")
    }
}

/// Headless preview: prefer a loopback device, fall back to a null sink.
fn build_sink() -> Box<dyn FrameSink> {
    #[cfg(feature = "v4l2")]
    {
        let resolution = Resolution {
            width: PREVIEW_WIDTH,
            height: PREVIEW_HEIGHT,
        };
        if let Ok(sink) = frameiru_sink::v4l2::LoopbackSink::open("/dev/video10", resolution) {
            return Box::new(sink);
        }
        eprintln!("no loopback device; running headless preview only");
    }
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
}
