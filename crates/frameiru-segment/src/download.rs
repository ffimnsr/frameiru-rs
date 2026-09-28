//! Model download helper (feature `onnx`).
//!
//! Fetches standard portrait-matting ONNX weights into the local models
//! directory so the pipeline can run without manual setup.

use std::io::Write;
use std::path::Path;

use frameiru_core::error::FrameiruError;

/// RMBG-2.0 (briaai) ONNX weights on Hugging Face.
pub const RMBG20_URL: &str = "https://huggingface.co/briaai/RMBG-2.0/resolve/main/onnx/model.onnx";

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
    fn rmbg20_url_is_https() {
        assert!(RMBG20_URL.starts_with("https://"));
        assert!(RMBG20_URL.ends_with(".onnx"));
    }

    #[test]
    fn download_fails_on_missing_destination_dir() {
        // /proc is read-only: creation must fail cleanly, not panic.
        let err = download_model(RMBG20_URL, Path::new("/proc/frameiru-model.onnx"));
        assert!(err.is_err());
    }
}
