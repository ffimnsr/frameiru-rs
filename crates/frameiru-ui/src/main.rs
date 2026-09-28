//! Frameiru GUI launcher.

mod app;

use std::path::PathBuf;

use app::{App, MainWindow};
use clap::Parser;
use frameiru_core::format::Resolution;

#[derive(Debug, Parser)]
#[command(name = "frameiru-ui", about = "Frameiru desktop GUI")]
struct Cli {
    /// ONNX segmentation model path (optional).
    #[arg(long)]
    model: Option<PathBuf>,
    /// ONNX model input canvas (WxH); must match the graph.
    #[arg(long, default_value = "320x320")]
    input_size: String,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let (w, h) = cli
        .input_size
        .split_once('x')
        .map(|(w, h)| (w.parse::<u32>(), h.parse::<u32>()))
        .and_then(|(w, h)| match (w, h) {
            (Ok(w), Ok(h)) if w > 0 && h > 0 => Some((w, h)),
            _ => None,
        })
        .unwrap_or_else(|| {
            eprintln!("bad --input-size {:?}; using 320x320", cli.input_size);
            (320, 320)
        });
    let model_input = Resolution {
        width: w,
        height: h,
    };
    let window = MainWindow::new()?;
    let app = App::new(window, cli.model, model_input);
    app.run()
}
