//! Central error type shared across the frameiru workspace.

use crate::format::PixelFormat;

/// Central error type for all frameiru crates.
#[derive(Debug, thiserror::Error)]
pub enum FrameiruError {
    /// The [`crate::BufferPool`] has no free buffers left.
    #[error("buffer pool exhausted (capacity {0})")]
    PoolExhausted(usize),

    /// An argument failed basic validation.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    /// Expected one pixel format but found another.
    #[error("pixel format mismatch: expected {expected:?}, got {actual:?}")]
    FormatMismatch {
        expected: PixelFormat,
        actual: PixelFormat,
    },

    /// A buffer is too small to hold the required frame data.
    #[error("buffer too small: need {need} bytes, have {have}")]
    InsufficientCapacity { need: usize, have: usize },

    /// An operation is not supported in the current build/config.
    #[error("unsupported operation: {0}")]
    Unsupported(String),

    /// Segmentation/inference stage failure.
    #[error("segmentation error: {0}")]
    Segmentation(String),

    /// Compositing stage failure.
    #[error("composition error: {0}")]
    Composition(String),

    /// Internal/plumbing failure (channels, lifecycle).
    #[error("internal error: {0}")]
    Internal(String),

    /// Wrapper around I/O errors.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}
