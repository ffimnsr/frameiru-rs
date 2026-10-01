//! Frameiru control plane: Unix domain socket server/client with a JSON
//! protocol, plus an optional D-Bus provider (feature `dbus`).

pub mod protocol;
pub mod socket;

#[cfg(feature = "dbus")]
pub mod dbus;

pub use protocol::{IpcRequest, IpcResponse, StatusInfo};
pub use socket::{IpcClient, IpcServer};

#[cfg(feature = "dbus")]
pub use dbus::serve as serve_dbus;

use frameiru_core::error::FrameiruError;
use frameiru_core::BackgroundMode;

/// Interface the control plane dispatches to — implemented by the pipeline
/// engine's handle (and by fakes in tests).
pub trait Control: Send + Sync + 'static {
    fn set_background(&self, mode: BackgroundMode) -> Result<(), FrameiruError>;
    fn status(&self) -> StatusInfo;
    fn stop(&self) -> Result<(), FrameiruError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::BufReader;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    struct FakeControl {
        background: Mutex<BackgroundMode>,
        stopped: AtomicBool,
    }

    impl Control for FakeControl {
        fn set_background(&self, mode: BackgroundMode) -> Result<(), FrameiruError> {
            *self.background.lock().unwrap() = mode;
            Ok(())
        }

        fn status(&self) -> StatusInfo {
            StatusInfo {
                running: !self.stopped.load(std::sync::atomic::Ordering::SeqCst),
                resolution: None,
                capture_fps: 10.0,
                composite_fps: 9.0,
                frames_composited: 7,
                composites_skipped: 2,
                masks_computed: 7,
                latency_us: 3,
                background: self.background.lock().unwrap().clone(),
            }
        }

        fn stop(&self) -> Result<(), FrameiruError> {
            self.stopped
                .store(true, std::sync::atomic::Ordering::SeqCst);
            Ok(())
        }
    }

    /// Unique socket path per test (pid + counter), per flaky-test rules.
    fn socket_path() -> PathBuf {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        std::env::temp_dir().join(format!(
            "frameiru-ipc-{}-{}.sock",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::SeqCst)
        ))
    }

    fn server_thread(path: PathBuf, control: Arc<dyn Control>) -> std::thread::JoinHandle<()> {
        let server = IpcServer::bind(&path).unwrap();
        std::thread::spawn(move || {
            server.run(control).unwrap();
        })
    }

    /// Waits for the server thread to bind `path` (bounded retry, no fixed
    /// sleep — the accept loop polls every 50ms so this is quick).
    fn wait_for_server(path: &std::path::Path) {
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while std::time::Instant::now() < deadline {
            if path.exists() {
                return;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        panic!("server did not bind {} in time", path.display());
    }

    #[test]
    fn set_background_and_get_status_roundtrip() {
        let path = socket_path();
        let control = Arc::new(FakeControl {
            background: Mutex::new(BackgroundMode::Passthrough),
            stopped: AtomicBool::new(false),
        });
        let thread = server_thread(path.clone(), Arc::clone(&control) as Arc<dyn Control>);
        wait_for_server(&path);

        let client = IpcClient::connect(&path).unwrap();
        client
            .set_background(BackgroundMode::Color { r: 9, g: 8, b: 7 })
            .unwrap();
        let status = client.status().unwrap();
        assert!(status.running);
        assert_eq!(
            status.background,
            BackgroundMode::Color { r: 9, g: 8, b: 7 }
        );
        assert_eq!(status.frames_composited, 7);

        client.stop().unwrap();
        thread.join().unwrap();
        assert!(!path.exists(), "socket must be unlinked after stop");
    }

    #[test]
    fn malformed_request_gets_error_response() {
        let path = socket_path();
        let control = Arc::new(FakeControl {
            background: Mutex::new(BackgroundMode::Passthrough),
            stopped: AtomicBool::new(false),
        });
        let thread = server_thread(path.clone(), Arc::clone(&control) as Arc<dyn Control>);
        wait_for_server(&path);

        let client = IpcClient::connect(&path).unwrap();
        let resp = client.request(&IpcRequest::GetStatus).unwrap();
        assert!(matches!(resp, IpcResponse::Status { .. }));
        // Raw garbage on the wire.
        {
            let stream = std::os::unix::net::UnixStream::connect(&path).unwrap();
            use std::io::Write;
            let mut stream = stream;
            stream.write_all(b"this is not json\n").unwrap();
            let mut line = String::new();
            use std::io::BufRead;
            let mut reader = BufReader::new(stream);
            reader.read_line(&mut line).unwrap();
            let parsed: IpcResponse = serde_json::from_str(&line).unwrap();
            assert!(matches!(parsed, IpcResponse::Error { .. }));
        }
        client.stop().unwrap();
        thread.join().unwrap();
    }

    #[test]
    fn stale_socket_is_replaced_on_bind() {
        let path = socket_path();
        std::fs::write(&path, b"stale").unwrap();
        let server = IpcServer::bind(&path).unwrap();
        assert!(path.exists());
        drop(server);
        assert!(!path.exists(), "drop must unlink the socket");
    }
}
