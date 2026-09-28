//! Zero-allocation frame buffers and the pool that recycles them.
//!
//! The hot path (capture -> segment -> composite -> sink) must not allocate.
//! [`BufferPool`] preallocates `capacity` frames up front; every
//! [`PooledBuffer`] hands its buffer back to the pool on drop so the backing
//! `Vec`s are reused without reallocation or zeroing.

use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex, MutexGuard};

use crate::error::FrameiruError;
use crate::format::{FrameMetadata, PixelFormat, Resolution};

/// Owned video frame: metadata plus tightly packed pixel data.
///
/// Cheap to clone: `data` is a shared `Vec` copy-on-write clone; frames that
/// outlive the producing stage (preview taps, sinks) should prefer
/// `Arc<FrameBuffer>` sharing instead.
#[derive(Debug, Clone)]
pub struct FrameBuffer {
    pub metadata: FrameMetadata,
    pub data: Vec<u8>,
}

impl FrameBuffer {
    /// Creates a frame with `data` sized exactly to its metadata.
    pub fn new(metadata: FrameMetadata) -> Self {
        let len = metadata.bytes_len();
        Self::with_capacity(metadata, len)
    }

    /// Creates a frame whose buffer is pre-sized to `capacity` bytes
    /// (typically `metadata.bytes_len()`) without extra reallocations.
    pub fn with_capacity(metadata: FrameMetadata, capacity: usize) -> Self {
        Self {
            metadata,
            data: vec![0; capacity],
        }
    }
}

/// Soft segmentation mask: `0.0` = background, `1.0` = foreground.
#[derive(Debug, Clone, Default)]
pub struct Mask {
    pub resolution: Resolution,
    pub data: Vec<f32>,
}

impl Mask {
    /// Fills a mask of `resolution` with a constant value.
    pub fn filled(resolution: Resolution, value: f32) -> Self {
        let len = (resolution.width * resolution.height) as usize;
        Self {
            resolution,
            data: vec![value; len],
        }
    }
}

/// Shared pool state, kept in an `Arc` so [`PooledBuffer`] can hand its frame
/// back on drop without borrowing the pool.
struct PoolState {
    free: Vec<FrameBuffer>,
    allocated: usize,
}

/// Preallocated pool of [`FrameBuffer`]s.
///
/// Frames are recycled: acquire a [`PooledBuffer`], use it, drop it, and the
/// buffer returns to the pool for reuse.
pub struct BufferPool {
    state: Arc<Mutex<PoolState>>,
    capacity: usize,
    resolution: Resolution,
    format: PixelFormat,
}

impl BufferPool {
    /// Preallocates `capacity` frames for `resolution`/`format`.
    pub fn new(
        resolution: Resolution,
        format: PixelFormat,
        capacity: usize,
    ) -> Result<Self, FrameiruError> {
        if !resolution.is_valid() {
            return Err(FrameiruError::InvalidArgument(
                "cannot build pool for invalid resolution".into(),
            ));
        }
        if capacity == 0 {
            return Err(FrameiruError::InvalidArgument(
                "pool capacity must be at least 1".into(),
            ));
        }

        let frame_size = format.bytes_per_frame(resolution);
        let metadata = FrameMetadata {
            sequence: 0,
            timestamp_us: 0,
            resolution,
            format,
        };
        let mut free = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            free.push(FrameBuffer::with_capacity(metadata, frame_size));
        }

        Ok(Self {
            state: Arc::new(Mutex::new(PoolState {
                free,
                allocated: capacity,
            })),
            capacity,
            resolution,
            format,
        })
    }

    /// Takes a buffer from the pool, enlarging it if preallocation hasn't
    /// happened yet. Fails with [`FrameiruError::PoolExhausted`] once
    /// `capacity` buffers are outstanding.
    pub fn acquire(&self) -> Result<PooledBuffer, FrameiruError> {
        let mut state = Self::lock(&self.state);
        let frame = if let Some(mut frame) = state.free.pop() {
            frame.metadata.resolution = self.resolution;
            frame.metadata.format = self.format;
            frame
        } else if state.allocated < self.capacity {
            state.allocated += 1;
            FrameBuffer::new(FrameMetadata {
                sequence: 0,
                timestamp_us: 0,
                resolution: self.resolution,
                format: self.format,
            })
        } else {
            return Err(FrameiruError::PoolExhausted(self.capacity));
        };
        Ok(PooledBuffer {
            pool: Arc::clone(&self.state),
            frame: Some(frame),
        })
    }

    /// Number of buffers currently resting in the pool and ready for reuse.
    pub fn available(&self) -> usize {
        Self::lock(&self.state).free.len()
    }

    /// Total number of buffers the pool has ever issued.
    pub fn allocated(&self) -> usize {
        Self::lock(&self.state).allocated
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn resolution(&self) -> Resolution {
        self.resolution
    }

    pub fn format(&self) -> PixelFormat {
        self.format
    }

    fn lock(state: &Arc<Mutex<PoolState>>) -> MutexGuard<'_, PoolState> {
        state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// A [`FrameBuffer`] borrowed from a [`BufferPool`].
///
/// When dropped, the frame is returned to the pool it came from. The buffer
/// shrinks back to pool size on the next acquire, so reallocation and zeroing
/// are avoided across a frame's lifetime.
pub struct PooledBuffer {
    pool: Arc<Mutex<PoolState>>,
    frame: Option<FrameBuffer>,
}

impl PooledBuffer {
    pub fn frame(&self) -> &FrameBuffer {
        self.frame.as_ref().expect("pooled buffer is present")
    }

    pub fn frame_mut(&mut self) -> &mut FrameBuffer {
        self.frame.as_mut().expect("pooled buffer is present")
    }
}

impl Deref for PooledBuffer {
    type Target = FrameBuffer;

    fn deref(&self) -> &FrameBuffer {
        self.frame()
    }
}

impl DerefMut for PooledBuffer {
    fn deref_mut(&mut self) -> &mut FrameBuffer {
        self.frame_mut()
    }
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        if let Some(frame) = self.frame.take() {
            if let Ok(mut state) = self.pool.lock() {
                state.free.push(frame);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: u32 = 64;
    const H: u32 = 48;

    fn pool() -> BufferPool {
        let res = Resolution {
            width: W,
            height: H,
        };
        BufferPool::new(res, PixelFormat::Rgb8, 3).unwrap()
    }

    fn resolution() -> Resolution {
        Resolution {
            width: W,
            height: H,
        }
    }

    #[test]
    fn rejects_invalid_config() {
        let res = resolution();
        assert!(BufferPool::new(res, PixelFormat::Rgb8, 0).is_err());
        assert!(BufferPool::new(
            Resolution {
                width: 0,
                height: H
            },
            PixelFormat::Rgb8,
            2
        )
        .is_err());
        assert!(BufferPool::new(
            Resolution {
                width: W,
                height: 0
            },
            PixelFormat::Rgb8,
            2
        )
        .is_err());
    }

    #[test]
    fn acquire_returns_expected_layout() {
        let p = pool();
        let buf = p.acquire().unwrap();
        assert_eq!(buf.metadata.resolution, p.resolution());
        assert_eq!(buf.metadata.format, p.format());
        assert_eq!(
            buf.data.len(),
            PixelFormat::Rgb8.bytes_per_frame(resolution())
        );
        assert_eq!(p.available(), 2);
        assert_eq!(p.allocated(), 3);
    }

    #[test]
    fn pool_exhausts_and_recycles() {
        let p = pool();
        let mut held = Vec::new();
        for _ in 0..p.capacity() {
            held.push(p.acquire().unwrap());
        }
        assert!(matches!(p.acquire(), Err(FrameiruError::PoolExhausted(3))));

        drop(held);
        assert_eq!(p.available(), 3);
        assert_eq!(p.allocated(), 3);
    }

    #[test]
    fn buffers_are_reused_without_realloc() {
        let p = pool();
        let buf = p.acquire().unwrap();
        let data_capacity = buf.data.capacity();
        assert_eq!(
            data_capacity,
            PixelFormat::Rgb8.bytes_per_frame(resolution())
        );
        drop(buf);

        let buf = p.acquire().unwrap();
        assert_eq!(buf.data.capacity(), data_capacity);
    }

    #[test]
    fn masked_filled_has_expected_length() {
        let mask = Mask::filled(resolution(), 1.0);
        assert_eq!(mask.data.len(), (W * H) as usize);
        assert!(mask.data.iter().all(|&v| v == 1.0));
    }
}
