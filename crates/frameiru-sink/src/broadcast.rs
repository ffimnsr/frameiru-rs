//! In-memory broadcast sink for UI previews and IPC monitors.
//!
//! Shares each frame as an `Arc<FrameBuffer>` with all subscribers, so
//! preview consumers read the same pixels without copying. `write_frame` is
//! synchronous; only `subscribe`/`recv` require a Tokio runtime.

use std::sync::Arc;

use frameiru_core::error::FrameiruError;
use frameiru_core::traits::FrameSink;
use frameiru_core::FrameBuffer;
use tokio::sync::broadcast;

/// Fan-out sink that clones `Arc<FrameBuffer>` to every subscriber.
///
/// The backing channel has fixed capacity; when consumers lag, the oldest
/// unread frame is dropped for the slowest receiver (standard Tokio
/// `broadcast` semantics, surfaced as `RecvError::Lagged`).
pub struct BroadcastSink {
    tx: broadcast::Sender<Arc<FrameBuffer>>,
    capacity: usize,
}

impl BroadcastSink {
    /// Create a sink with a ring buffer holding up to `capacity` frames.
    ///
    /// Capacity must be at least 1. A capacity around 2-4 smooths preview
    /// jitter without adding meaningful latency.
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            tx: broadcast::channel(capacity).0,
            capacity,
        }
    }

    /// Subscribe to the preview stream.
    pub fn subscribe(&self) -> broadcast::Receiver<Arc<FrameBuffer>> {
        self.tx.subscribe()
    }

    /// Number of frames the ring buffer can hold before lagging.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Current number of subscribers.
    pub fn subscriber_count(&self) -> usize {
        self.tx.receiver_count()
    }
}

impl FrameSink for BroadcastSink {
    fn write_frame(&mut self, frame: &FrameBuffer) -> Result<(), FrameiruError> {
        // With no subscribers the send fails; that is not an error — the
        // pipeline just skips an unobserved preview frame.
        let _ = self.tx.send(Arc::new(frame.clone()));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use frameiru_core::format::{FrameMetadata, PixelFormat, Resolution};

    fn frame(sequence: u64) -> FrameBuffer {
        FrameBuffer {
            metadata: FrameMetadata {
                sequence,
                timestamp_us: 0,
                resolution: Resolution {
                    width: 2,
                    height: 2,
                },
                format: PixelFormat::Rgb8,
            },
            data: vec![sequence as u8; 12],
        }
    }

    #[test]
    fn subscriber_receives_shared_frame() {
        let mut sink = BroadcastSink::new(4);
        let mut rx = sink.subscribe();
        sink.write_frame(&frame(7)).unwrap();
        let got = rx.try_recv().unwrap();
        assert_eq!(got.metadata.sequence, 7);
        assert_eq!(got.data, vec![7u8; 12]);
    }

    #[test]
    fn all_subscribers_receive_each_frame() {
        let mut sink = BroadcastSink::new(4);
        let mut rx1 = sink.subscribe();
        let mut rx2 = sink.subscribe();
        sink.write_frame(&frame(1)).unwrap();
        sink.write_frame(&frame(2)).unwrap();
        assert_eq!(rx1.try_recv().unwrap().metadata.sequence, 1);
        assert_eq!(rx1.try_recv().unwrap().metadata.sequence, 2);
        assert_eq!(rx2.try_recv().unwrap().metadata.sequence, 1);
        assert_eq!(rx2.try_recv().unwrap().metadata.sequence, 2);
    }

    #[test]
    fn write_without_subscribers_is_not_an_error() {
        let mut sink = BroadcastSink::new(2);
        sink.write_frame(&frame(0)).unwrap();
        assert_eq!(sink.subscriber_count(), 0);
    }

    #[test]
    fn lagging_subscriber_reports_skipped_frames() {
        let mut sink = BroadcastSink::new(2);
        let mut rx = sink.subscribe();
        for seq in 0..5 {
            sink.write_frame(&frame(seq)).unwrap();
        }
        match rx.try_recv() {
            Err(tokio::sync::broadcast::error::TryRecvError::Lagged(skipped)) => {
                assert!(skipped >= 3, "expected at least 3 skipped frames");
            }
            other => panic!("expected Lagged, got {other:?}"),
        }
        // Recovers: next recv yields an unread retained frame.
        let got = rx.try_recv().unwrap();
        assert!(got.metadata.sequence >= 3);
    }

    #[test]
    fn capacity_is_at_least_one() {
        assert_eq!(BroadcastSink::new(0).capacity(), 1);
        assert_eq!(BroadcastSink::new(3).capacity(), 3);
    }
}
