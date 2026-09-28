//! Model registry and download helper (feature `onnx`).
//!
//! All models here are hosted as GitHub release assets by the `rembg`
//! project: direct downloads, no Hugging Face account or token required.
//! (The upstream BRIA RMBG-2.0 repo is gated; its weights are mirrored
//! here.)

use std::io::Write;
use std::path::Path;

use frameiru_core::error::FrameiruError;
use frameiru_core::format::Resolution;

/// A downloadable segmentation model and its inference input geometry.
#[derive(Debug, Clone, Copy)]
pub struct ModelSpec {
    pub name: &'static str,
    pub url: &'static str,
    /// Fixed input canvas the ONNX graph expects.
    pub input: Resolution,
    /// Approximate download size in MiB.
    pub size_mb: u64,
    pub note: &'static str,
}

/// DIS (xuebinqin) portrait matting: best quality/size balance for a webcam.
pub const SILUETA: ModelSpec = ModelSpec {
    name: "silueta",
    url: "https://github.com/danielgatis/rembg/releases/download/v0.0.0/silueta.onnx",
    input: Resolution {
        width: 320,
        height: 320,
    },
    size_mb: 42,
    note: "good quality/size balance; default",
};

/// Lightweight U2-Net variant: fastest, noticeably lower quality.
pub const U2NETP: ModelSpec = ModelSpec {
    name: "u2netp",
    url: "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx",
    input: Resolution {
        width: 320,
        height: 320,
    },
    size_mb: 4,
    note: "fastest/lightest; quick smoke tests",
};

/// Classic U2-Net.
pub const U2NET: ModelSpec = ModelSpec {
    name: "u2net",
    url: "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx",
    input: Resolution {
        width: 320,
        height: 320,
    },
    size_mb: 167,
    note: "classic U2-Net",
};

/// isnet-general-use: high quality at 1024px input.
pub const ISNET: ModelSpec = ModelSpec {
    name: "isnet-general-use",
    url: "https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx",
    input: Resolution {
        width: 1024,
        height: 1024,
    },
    size_mb: 170,
    note: "high quality; 1024px input is slower",
};

/// RMBG-2.0 (BRIA) mirrored from the gated Hugging Face repo.
pub const RMBG20_MIRROR: ModelSpec = ModelSpec {
    name: "rmbg-2.0",
    url: "https://github.com/danielgatis/rembg/releases/download/v0.0.0/bria-rmbg-2.0.onnx",
    input: Resolution {
        width: 1024,
        height: 1024,
    },
    size_mb: 976,
    note: "RMBG-2.0 via mirror (HF original requires sign-in); heavy",
};

/// BiRefNet portrait matting: top quality, very heavy.
pub const BIREFNET_PORTRAIT: ModelSpec = ModelSpec {
    name: "birefnet-portrait",
    url: "https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-portrait-epoch_150.onnx",
    input: Resolution { width: 1024, height: 1024 },
    size_mb: 927,
    note: "best portrait quality; very heavy",
};

/// All known models, in download-recommendation order.
pub const MODELS: &[ModelSpec] = &[
    SILUETA,
    U2NETP,
    U2NET,
    ISNET,
    RMBG20_MIRROR,
    BIREFNET_PORTRAIT,
];

/// The model `frameiru models download` fetches when nothing is specified.
pub fn default_model() -> &'static ModelSpec {
    &SILUETA
}

/// Looks up a model by name (case-insensitive).
pub fn find_model(name: &str) -> Option<&'static ModelSpec> {
    MODELS.iter().find(|m| m.name.eq_ignore_ascii_case(name))
}

/// Downloads `url` to `dest`, overwriting any existing file.
pub fn download_model(url: &str, dest: &Path) -> Result<(), FrameiruError> {
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            FrameiruError::Segmentation(format!("cannot create {}: {e}", parent.display()))
        })?;
    }
    let response = ureq::get(url)
        .call()
        .map_err(|e| FrameiruError::Segmentation(format!("download {url} failed: {e}")))?;

    let mut reader = response.into_reader();
    let mut file = std::fs::File::create(dest).map_err(|e| {
        FrameiruError::Segmentation(format!("cannot create {}: {e}", dest.display()))
    })?;
    let copied = std::io::copy(&mut reader, &mut file)
        .map_err(|e| FrameiruError::Segmentation(format!("download {url} failed: {e}")))?;
    file.flush().map_err(|e| {
        FrameiruError::Segmentation(format!("flush {} failed: {e}", dest.display()))
    })?;
    if copied == 0 {
        return Err(FrameiruError::Segmentation(format!(
            "download {url} produced an empty file"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_entries_are_sane() {
        assert!(!MODELS.is_empty());
        for model in MODELS {
            assert!(model.url.starts_with("https://"), "{}", model.name);
            assert!(model.url.ends_with(".onnx"), "{}", model.name);
            assert!(model.input.is_valid(), "{}", model.name);
            assert!(model.size_mb > 0, "{}", model.name);
        }
    }

    #[test]
    fn default_and_lookup_work() {
        assert_eq!(default_model().name, "silueta");
        assert_eq!(find_model("SILUETA").map(|m| m.name), Some("silueta"));
        assert_eq!(
            find_model("isnet-general-use").map(|m| m.size_mb),
            Some(170)
        );
        assert!(find_model("does-not-exist").is_none());
    }

    #[test]
    fn download_fails_on_missing_destination_dir() {
        // /proc is read-only: creation must fail cleanly, not panic.
        let err = download_model(default_model().url, Path::new("/proc/frameiru-model.onnx"));
        assert!(err.is_err());
    }
}
