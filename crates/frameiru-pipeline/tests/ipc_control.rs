//! End-to-end control plane: engine + IPC server + client.

use std::path::PathBuf;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::time::Duration;

use frameiru_capture::MockSource;
use frameiru_compose::CpuCompositor;
use frameiru_core::{BackgroundMode, Resolution};
use frameiru_ipc::{Control, IpcClient, IpcServer};
use frameiru_pipeline::{Engine, PipelineConfig};
use frameiru_sink::MockSink;

fn res() -> Resolution {
    Resolution {
        width: 64,
        height: 48,
    }
}

fn socket_path() -> PathBuf {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    std::env::temp_dir().join(format!(
        "frameiru-pipeline-ipc-{}-{}.sock",
        std::process::id(),
        COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
    ))
}

/// Full loop: engine runs, IPC server exposes it, client changes the
/// background, reads status, and stops everything over the socket.
#[test]
fn control_the_engine_over_the_socket() {
    let source = MockSource::new(res()).unwrap();
    let compositor = CpuCompositor::new();
    let engine = Engine::start(
        PipelineConfig {
            max_fps: 0,
            ..Default::default()
        },
        Box::new(source),
        Some(Box::new(ZeroSegmenter)),
        Box::new(compositor),
        Box::new(MockSink::new()),
    )
    .unwrap();
    let handle = engine.handle();

    let path = socket_path();
    let server = IpcServer::bind(&path).unwrap();
    let control: Arc<dyn Control> = Arc::new(handle.clone());
    let server_thread = std::thread::spawn(move || server.run(control).unwrap());
    std::thread::sleep(Duration::from_millis(50)); // let the server bind

    let client = IpcClient::connect(&path).unwrap();

    // Status reports a running pipeline with a valid resolution.
    let status = client.status().unwrap();
    assert!(status.running);
    assert_eq!(status.resolution, Some(res()));

    // Switch background over IPC; status follows the applied mode.
    client
        .set_background(BackgroundMode::Color { r: 255, g: 0, b: 0 })
        .unwrap();
    wait_until("red background applied", || {
        client.status().unwrap().background == BackgroundMode::Color { r: 255, g: 0, b: 0 }
    });
    client
        .set_background(BackgroundMode::Color { r: 0, g: 0, b: 255 })
        .unwrap();
    wait_until("blue background applied", || {
        client.status().unwrap().background == BackgroundMode::Color { r: 0, g: 0, b: 255 }
    });

    // Stop over IPC: engine shuts down, server unlinks the socket.
    client.stop().unwrap();
    server_thread.join().unwrap();
    assert!(!path.exists(), "socket unlinked after stop");
    assert!(!handle.is_running());
}

/// Mask 0: output is purely the background color.
struct ZeroSegmenter;

impl frameiru_core::traits::Segmenter for ZeroSegmenter {
    fn input_resolution(&self) -> Resolution {
        res()
    }

    fn segment(
        &mut self,
        frame: &frameiru_core::FrameBuffer,
    ) -> Result<frameiru_core::buffer::Mask, frameiru_core::error::FrameiruError> {
        Ok(frameiru_core::buffer::Mask::filled(
            frame.metadata.resolution,
            0.0,
        ))
    }
}

fn wait_until(what: &str, pred: impl Fn() -> bool) {
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while std::time::Instant::now() < deadline {
        if pred() {
            return;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    panic!("timed out waiting for {what}");
}
