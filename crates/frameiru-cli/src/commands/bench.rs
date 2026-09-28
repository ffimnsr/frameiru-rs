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
    /// Background used for compositing.
    #[arg(long, default_value = "color:0,120,0")]
    pub background: String,
}

pub fn benchmark(args: BenchArgs) -> anyhow::Result<()> {
    let resolution = super::parse_resolution(&args.resolution)?;
    let background = super::parse_background(&args.background)?;

    let source = MockSource::new(resolution)?;
    let segmenter: Option<Box<dyn Segmenter>> = match &args.model {
        Some(path) => {
            #[cfg(feature = "onnx")]
            {
                let config = frameiru_segment::OnnxConfig::new(resolution)?;
                Some(Box::new(frameiru_segment::OnnxSegmenter::load(
                    path, config,
                )?))
            }
            #[cfg(not(feature = "onnx"))]
            {
                let _ = path;
                anyhow::bail!(
                    "segmentation needs the `onnx` feature; rebuild with `cargo build --features onnx`"
                )
            }
        }
        None => None,
    };

    let mut compositor = CpuCompositor::new();
    compositor.update_background(background.clone())?;

    let engine = Engine::start(
        PipelineConfig {
            max_fps: 0,
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
    println!("latency     : {} us (last frame)", metrics.latency_us);
    println!("mask errors : {}", metrics.mask_errors);
    Ok(())
}

fn print_progress(done: u64, total: u64) {
    if done.is_multiple_of(50) || done == total {
        eprintln!("  {done}/{total} frames");
    }
}
