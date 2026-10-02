//! `benchmark`: throughput/latency numbers without a video sink.

use clap::Args;
use frameiru_capture::MockSource;
use frameiru_compose::CpuCompositor;
use frameiru_core::traits::{Compositor, Segmenter};
use frameiru_pipeline::{Engine, PipelineConfig};
use frameiru_sink::MockSink;

#[derive(Debug, Clone, Args)]
pub struct BenchArgs {
    /// Number of frames to compose.
    #[arg(long, default_value_t = 300)]
    pub frames: u64,
    /// Capture resolution.
    #[arg(long, default_value = "640x480")]
    pub resolution: String,
    /// ONNX model path; benchmark segmentation when set.
    #[arg(long)]
    pub model: Option<std::path::PathBuf>,
    /// ONNX model input canvas (WxH) when `--model` is given; the embedded
    /// default is 256x256.
    #[arg(long, default_value = "256x256")]
    pub input_size: String,
    /// Tensor normalization: `imagenet` (default) or `unit` (MediaPipe).
    #[arg(long, default_value = "imagenet")]
    pub normalization: String,
    /// Mask EMA blending factor: `off`, or 0.0 (freeze) ..= 1.0.
    #[arg(long, default_value = "off")]
    pub mask_alpha: String,
    /// Dynamic crop & track (ROI zoom) around the subject.
    #[arg(long)]
    pub roi_zoom: bool,
    /// Background used for compositing.
    #[arg(long, default_value = "color:0,120,0")]
    pub background: String,
    /// ORT intra-op threads; default pins to the physical core count.
    #[arg(long, value_name = "N")]
    pub threads: Option<usize>,
}

pub fn benchmark(args: BenchArgs) -> anyhow::Result<()> {
    let resolution = super::parse_resolution(&args.resolution)?;
    let background = super::parse_background(&args.background)?;

    let source = MockSource::new(resolution)?;
    let segmenter: Option<Box<dyn Segmenter>> = {
        #[cfg(feature = "onnx")]
        {
            use frameiru_segment::{load_embedded, load_model, OnnxConfig, EMBEDDED_MODEL_INPUT};
            let mut config = match &args.model {
                Some(_) => {
                    let input_size = super::parse_resolution(&args.input_size)?;
                    let mut config = OnnxConfig::new(input_size)?;
                    config.normalization = super::parse_normalization(&args.normalization)?;
                    config
                }
                // No `--model`: benchmark the embedded fusion model.
                None => {
                    let mut config = OnnxConfig::new(EMBEDDED_MODEL_INPUT)?;
                    config.normalization = frameiru_segment::embedded_normalization();
                    config
                }
            };
            // Benchmark the raw model: guided refinement and mask polish are
            // fixed post-steps that would muddy masks/s comparisons.
            config.refine_mask = false;
            config.mask_dilate = 0;
            config.mask_contrast = 0.0;
            config.roi_zoom = args.roi_zoom;
            if args.threads == Some(0) {
                anyhow::bail!("--threads must be >= 1 (omit it for the physical-core default)");
            }
            config.intra_threads = args.threads;
            let segmenter: Box<dyn Segmenter> = match &args.model {
                Some(path) => load_model(path, config)?,
                None => load_embedded(config)?,
            };
            Some(segmenter)
        }
        #[cfg(not(feature = "onnx"))]
        {
            if args.model.is_some() {
                anyhow::bail!(
                    "segmentation needs the `onnx` feature; rebuild with `cargo build --features onnx`"
                )
            }
            None
        }
    };

    let mut compositor = CpuCompositor::new();
    compositor.update_background(background.clone())?;

    let engine = Engine::start(
        PipelineConfig {
            max_fps: 0,
            mask_alpha: super::parse_mask_alpha(&args.mask_alpha)?,
            // Benchmarks gate raw model throughput: never throttle inference.
            infer_max_fps: 0,
            ..Default::default()
        },
        Box::new(source),
        segmenter,
        Box::new(compositor),
        Box::new(MockSink::new()),
    )?;
    let handle = engine.handle();

    let started = std::time::Instant::now();
    let mut last = 0u64;
    while handle.metrics().frames_composited < args.frames {
        std::thread::sleep(std::time::Duration::from_millis(20));
        let now = handle.metrics().frames_composited;
        if now != last {
            last = now;
            print_progress(now, args.frames);
        }
        if started.elapsed() > std::time::Duration::from_secs(120) {
            anyhow::bail!(
                "benchmark timed out after {} of {} frames",
                now,
                args.frames
            );
        }
    }
    let elapsed = started.elapsed();
    let metrics = handle.metrics();
    engine.shutdown();

    println!("\nresolution  : {}x{}", resolution.width, resolution.height);
    println!("background  : {background:?}");
    println!(
        "frames      : {} composited / {} captured ({} dropped)",
        metrics.frames_composited, metrics.frames_captured, metrics.frames_dropped
    );
    println!(
        "throughput  : {:.1} composited fps (wall {:.2}s)",
        metrics.frames_composited as f64 / elapsed.as_secs_f64(),
        elapsed.as_secs_f64()
    );
    println!("capture fps : {:.1} (rolling)", metrics.capture_fps);
    println!(
        "masks       : {} computed in {:.2}s ({:.2} masks/s)",
        metrics.masks_computed,
        elapsed.as_secs_f64(),
        metrics.masks_computed as f64 / elapsed.as_secs_f64()
    );
    println!("latency     : {} us (last frame)", metrics.latency_us);
    println!("mask errors : {}", metrics.mask_errors);
    Ok(())
}

fn print_progress(done: u64, total: u64) {
    if done.is_multiple_of(50) || done == total {
        eprintln!("  {done}/{total} frames");
    }
}
