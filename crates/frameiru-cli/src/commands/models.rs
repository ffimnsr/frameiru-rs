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
    /// List known models, input sizes, and local presence.
    List,
    /// Download a model to disk. Defaults to `silueta` (no account needed;
    /// the gated RMBG-2.0 Hugging Face repo is mirrored, but 976 MB).
    Download {
        /// Model name from `models list` (or any direct .onnx URL with
        /// `--url`).
        #[arg(long)]
        model: Option<String>,
        /// Direct download URL; overrides `--model`.
        #[arg(long)]
        url: Option<String>,
        /// Destination file. Defaults to `models/<model>.onnx` when
        /// `--model` is given, otherwise `models/silueta.onnx`.
        #[arg(long)]
        output: Option<PathBuf>,
    },
}

pub fn models(cmd: ModelsCmd) -> anyhow::Result<()> {
    match cmd.command {
        ModelsSub::List => list(),
        ModelsSub::Download { model, url, output } => download(model, url, output),
    }
}

fn list() -> anyhow::Result<()> {
    #[cfg(feature = "onnx")]
    {
        for m in frameiru_segment::download::MODELS {
            let present = local_path_for(m.name).exists();
            println!(
                "{:<16} {:>4}x{:<4} {:>4} MB  {}  [{}]",
                m.name,
                m.input.width,
                m.input.height,
                m.size_mb,
                m.note,
                if present { "present" } else { "missing" }
            );
        }
        println!();
        println!(
            "default: {} ({})\nuse with: frameiru run --model <path> --input-size WxH",
            default_model().name,
            default_model().url
        );
        Ok(())
    }
    #[cfg(not(feature = "onnx"))]
    {
        eprintln!(
            "model listing needs the `onnx` feature; rebuild with `cargo build --features onnx`"
        );
        Ok(())
    }
}

fn download(
    model: Option<String>,
    url: Option<String>,
    output: Option<PathBuf>,
) -> anyhow::Result<()> {
    #[cfg(feature = "onnx")]
    {
        use frameiru_segment::download as dl;

        // Resolve the registry entry first so both URL and default output
        // follow the selected model.
        let spec = match model.as_deref().map(dl::find_model) {
            Some(Some(spec)) => Some(spec),
            Some(None) => anyhow::bail!(
                "unknown model {:?}; see `frameiru models list` for names",
                model.as_deref().unwrap_or("")
            ),
            None => None,
        };
        let (url, input_hint) = match url {
            Some(url) => (url, spec.map(|s| s.input)),
            None => (
                spec.map(|s| s.url)
                    .unwrap_or_else(|| dl::default_model().url)
                    .to_string(),
                Some(spec.map(|s| s.input).unwrap_or(dl::default_model().input)),
            ),
        };
        let name = spec.map(|s| s.name).unwrap_or(dl::default_model().name);
        let output = output.unwrap_or_else(|| PathBuf::from("models").join(format!("{name}.onnx")));
        if let Some(input) = input_hint {
            println!(
                "model input: {}x{} (pass this as --input-size)",
                input.width, input.height
            );
        }
        println!("downloading {url} -> {}", output.display());
        dl::download_model(&url, &output).with_context(|| format!("download failed ({url})"))?;
        println!(
            "done: {} ({} bytes)",
            output.display(),
            output.metadata()?.len()
        );
        Ok(())
    }
    #[cfg(not(feature = "onnx"))]
    {
        let _ = (model, url, output);
        bail!("model downloads need the `onnx` feature; rebuild with `cargo build --features onnx`")
    }
}

#[cfg(feature = "onnx")]
fn default_model() -> &'static frameiru_segment::download::ModelSpec {
    frameiru_segment::download::default_model()
}

#[cfg(feature = "onnx")]
fn local_path_for(name: &str) -> PathBuf {
    PathBuf::from("models").join(format!("{name}.onnx"))
}
