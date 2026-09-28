//! `models`: download and inspect ONNX segmentation weights.

use std::path::PathBuf;

#[cfg(not(feature = "onnx"))]
use anyhow::bail;
#[cfg(feature = "onnx")]
use anyhow::Context as _;
use clap::{Args, Subcommand};

#[derive(Debug, Clone, Args)]
pub struct ModelsCmd {
    #[command(subcommand)]
    pub command: ModelsSub,
}

#[derive(Debug, Clone, Subcommand)]
pub enum ModelsSub {
    /// List known models and whether they are present locally.
    List,
    /// Download a model (default: RMBG-2.0) to disk.
    Download {
        /// Model URL; defaults to the bundled RMBG-2.0 URL.
        #[arg(long)]
        url: Option<String>,
        /// Destination file.
        #[arg(long, default_value = "models/rmbg-2.0.onnx")]
        output: PathBuf,
    },
}

pub fn models(cmd: ModelsCmd) -> anyhow::Result<()> {
    match cmd.command {
        ModelsSub::List => list(),
        ModelsSub::Download { url, output } => download(url, output),
    }
}

fn list() -> anyhow::Result<()> {
    let default_path = PathBuf::from("models/rmbg-2.0.onnx");
    println!("name       : RMBG-2.0 (briaai)");
    #[cfg(feature = "onnx")]
    println!("url        : {}", frameiru_segment::download::RMBG20_URL);
    #[cfg(not(feature = "onnx"))]
    println!("url        : (rebuild with `--features onnx` to download)");
    println!(
        "local      : {} ({})",
        default_path.display(),
        if default_path.exists() {
            "present"
        } else {
            "missing"
        }
    );
    println!();
    println!("download with: frameiru models download");
    Ok(())
}

fn download(url: Option<String>, output: PathBuf) -> anyhow::Result<()> {
    #[cfg(feature = "onnx")]
    {
        let url = url.unwrap_or_else(|| frameiru_segment::download::RMBG20_URL.into());
        println!("downloading {url} -> {}", output.display());
        frameiru_segment::download::download_model(&url, &output)
            .with_context(|| format!("download failed ({url})"))?;
        println!(
            "done: {} ({} bytes)",
            output.display(),
            output.metadata()?.len()
        );
        Ok(())
    }
    #[cfg(not(feature = "onnx"))]
    {
        let _ = (url, output);
        bail!("model downloads need the `onnx` feature; rebuild with `cargo build --features onnx`")
    }
}
