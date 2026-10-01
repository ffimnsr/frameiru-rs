//! `run` (foreground) and `start` (daemon) commands.

use std::path::PathBuf;
use std::sync::Arc;

#[cfg_attr(all(feature = "v4l2", feature = "onnx"), allow(unused_imports))]
use anyhow::bail;
use anyhow::Context as _;
use frameiru_compose::new_compositor;
use frameiru_core::traits::{FrameSink, FrameSource, Segmenter};
use frameiru_ipc::{Control, IpcServer};
use frameiru_pipeline::{Engine, PipelineConfig};

use super::{parse_background, RunArgs};

/// Runs the full pipeline in the foreground until Ctrl-C or an IPC stop.
pub fn run(args: RunArgs) -> anyhow::Result<()> {
    let resolution = args.resolution();
    let background = parse_background(&args.background)?;

    let source: Box<dyn FrameSource> = build_source(&args)?;
    let segmenter: Option<Box<dyn Segmenter>> = build_segmenter(&args)?;
    let sink: Box<dyn FrameSink> = build_sink(&args)?;

    let compositor = new_compositor();

    let engine = Engine::start(
        PipelineConfig {
            max_fps: args.max_fps,
            mask_alpha: super::parse_mask_alpha(&args.mask_alpha)?,
            infer_max_fps: args.infer_fps,
            subject_light: super::parse_subject_light(&args.subject_light)?,
            ..Default::default()
        },
        source,
        segmenter,
        compositor,
        sink,
    )?;
    let handle = engine.handle();
    // Route the initial background through the engine so the applied mode is
    // tracked (status) and any invalid image path fails here, not mid-run.
    handle.update_background(background.clone())?;

    if let Some(socket) = &args.socket {
        let control: Arc<dyn Control> = Arc::new(handle.clone());
        let server = IpcServer::bind(socket)
            .with_context(|| format!("cannot bind control socket {}", socket.display()))?;
        std::thread::Builder::new()
            .name("frameiru-ipc".into())
            .spawn(move || {
                if let Err(e) = server.run(control) {
                    eprintln!("ipc server error: {e}");
                }
            })
            .expect("spawn ipc server thread");
        println!("control socket: {}", socket.display());
    }

    // Turn Ctrl-C into a clean engine shutdown.
    let ctrlc_handle = handle.clone();
    ctrlc::set_handler(move || ctrlc_handle.shutdown()).context("installing Ctrl-C handler")?;

    println!(
        "frameiru: {}x{} at {}/s, background {background:?}, output {}",
        resolution.width,
        resolution.height,
        if args.max_fps == 0 {
            "unlimited".to_string()
        } else {
            args.max_fps.to_string()
        },
        args.output.display()
    );
    while handle.is_running() {
        std::thread::sleep(std::time::Duration::from_millis(500));
    }
    Ok(())
}

/// `start` relaunches `run` as a detached background daemon with a control
/// socket and a log file, then returns.
pub fn start(args: RunArgs) -> anyhow::Result<()> {
    let socket = args.socket.clone().unwrap_or_else(super::default_socket);

    let log_path = log_path_for(&socket);
    let log_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .with_context(|| format!("cannot open log {}", log_path.display()))?;

    let mut command = std::process::Command::new(std::env::current_exe()?);
    // Detach into a new session so the daemon outlives the terminal.
    use std::os::unix::process::CommandExt;
    command
        .arg("run")
        .arg("--device")
        .arg(&args.device)
        .arg("--output")
        .arg(&args.output)
        .arg("--width")
        .arg(args.width.to_string())
        .arg("--height")
        .arg(args.height.to_string())
        .arg("--background")
        .arg(&args.background)
        .arg("--max-fps")
        .arg(args.max_fps.to_string())
        .arg("--socket")
        .arg(&socket)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::from(log_file.try_clone()?))
        .stderr(log_file)
        .process_group(0);
    if args.mock {
        command.arg("--mock");
    }
    if args.null_sink {
        command.arg("--null-sink");
    }
    if let Some(model) = &args.model {
        command.arg("--model").arg(model);
    }

    let child = command
        .spawn()
        .context("spawning daemon process (run `frameiru run` in the foreground instead)")?;

    let pid_path = pid_path_for(&socket);
    std::fs::write(&pid_path, child.id().to_string())
        .with_context(|| format!("cannot write pid file {}", pid_path.display()))?;

    println!(
        "frameiru started: pid {}, socket {}, log {}",
        child.id(),
        socket.display(),
        log_path.display()
    );
    println!("stop with: frameiru stop --socket {}", socket.display());
    Ok(())
}

fn build_source(args: &RunArgs) -> anyhow::Result<Box<dyn FrameSource>> {
    if args.mock {
        let source = frameiru_capture::MockSource::new(args.resolution())?;
        return Ok(Box::new(source));
    }
    #[cfg(feature = "v4l2")]
    {
        let source = frameiru_capture::v4l2::V4l2Source::open_with_resolution(
            &args.device,
            Some(args.resolution()),
        )
        .with_context(|| format!("cannot open capture device {}", args.device.display()))?;
        Ok(Box::new(source))
    }
    #[cfg(not(feature = "v4l2"))]
    bail!(
        "real camera capture needs the `v4l2` feature; rebuild with \
         `cargo build --features v4l2` or use `--mock`"
    )
}

fn build_segmenter(args: &RunArgs) -> anyhow::Result<Option<Box<dyn Segmenter>>> {
    #[cfg(feature = "onnx")]
    {
        use frameiru_segment::{load_embedded, load_model, OnnxConfig, EMBEDDED_MODEL_INPUT};
        let mut config = match &args.model {
            Some(path) => {
                let input_size = super::parse_resolution(&args.input_size)?;
                let mut config = OnnxConfig::new(input_size)?;
                config.normalization = super::parse_normalization(&args.normalization)?;
                config.mask_dilate = args.mask_dilate;
                config.mask_contrast = super::parse_mask_contrast(&args.mask_contrast)?;
                if args.threads == Some(0) {
                    anyhow::bail!("--threads must be >= 1 (omit it for the physical-core default)");
                }
                config.intra_threads = args.threads;
                let segmenter = load_model(path, config)?;
                return Ok(Some(segmenter));
            }
            // No `--model`: the embedded fusion model (MediaPipe + RVM) ships in the
            // binary (256x256, imagenet normalization) — zero-setup masks.
            None => {
                let mut config = OnnxConfig::new(EMBEDDED_MODEL_INPUT)?;
                config.normalization = frameiru_segment::embedded_normalization();
                config
            }
        };
        if args.threads == Some(0) {
            anyhow::bail!("--threads must be >= 1 (omit it for the physical-core default)");
        }
        config.intra_threads = args.threads;
        config.refine_mask = !args.no_refine_mask;
        config.mask_dilate = args.mask_dilate;
        config.mask_contrast = super::parse_mask_contrast(&args.mask_contrast)?;
        Ok(Some(load_embedded(config)?))
    }
    #[cfg(not(feature = "onnx"))]
    {
        if args.model.is_some() {
            bail!(
                "segmentation needs the `onnx` feature; rebuild with `cargo build --features onnx`"
            )
        }
        Ok(None)
    }
}

fn build_sink(args: &RunArgs) -> anyhow::Result<Box<dyn FrameSink>> {
    if args.null_sink {
        return Ok(Box::new(frameiru_sink::MockSink::new()));
    }
    #[cfg(feature = "v4l2")]
    {
        let sink = frameiru_sink::v4l2::LoopbackSink::open(&args.output, args.resolution())
            .with_context(|| format!("cannot open output device {}", args.output.display()))?;
        Ok(Box::new(sink))
    }
    #[cfg(not(feature = "v4l2"))]
    {
        let _ = args;
        bail!(
            "loopback output needs the `v4l2` feature; rebuild with `cargo build --features v4l2`"
        )
    }
}

fn log_path_for(socket: &std::path::Path) -> PathBuf {
    let mut path = socket.as_os_str().to_owned();
    path.push(".log");
    PathBuf::from(path)
}

fn pid_path_for(socket: &std::path::Path) -> PathBuf {
    let mut path = socket.as_os_str().to_owned();
    path.push(".pid");
    PathBuf::from(path)
}
