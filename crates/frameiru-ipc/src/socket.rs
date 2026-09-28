//! Unix domain socket server and client for the control plane.

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use frameiru_core::error::FrameiruError;
use frameiru_core::BackgroundMode;

use crate::protocol::{decode, encode, IpcRequest, IpcResponse, StatusInfo};
use crate::Control;

/// Poll interval for the non-blocking accept loop.
const ACCEPT_POLL: Duration = Duration::from_millis(50);
/// Read timeout for client connections; a stuck peer fails instead of
/// hanging the handler thread.
const CONN_TIMEOUT: Duration = Duration::from_secs(5);

/// Control-plane server bound to a Unix domain socket path.
///
/// `run` accepts connections and dispatches [`IpcRequest`]s to the provided
/// [`Control`]. A `Stop` request responds first, then stops the accept loop
/// and unlinks the socket file.
pub struct IpcServer {
    listener: UnixListener,
    path: PathBuf,
}

impl IpcServer {
    /// Binds `path`, removing a stale socket file left by a dead server.
    pub fn bind(path: impl AsRef<Path>) -> Result<Self, FrameiruError> {
        let path = path.as_ref().to_path_buf();
        if path.exists() {
            std::fs::remove_file(&path).map_err(|e| {
                FrameiruError::Io(std::io::Error::new(
                    e.kind(),
                    format!("cannot remove stale socket {}: {e}", path.display()),
                ))
            })?;
        }
        let listener = UnixListener::bind(&path)?;
        Ok(Self { listener, path })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Accepts and serves connections until a `Stop` request arrives (or the
    /// control's `stop` is invoked), then unlinks the socket. Blocks.
    pub fn run(&self, control: Arc<dyn Control>) -> Result<(), FrameiruError> {
        let stop = Arc::new(AtomicBool::new(false));
        self.listener.set_nonblocking(true)?;
        while !stop.load(Ordering::Relaxed) {
            match self.listener.accept() {
                Ok((stream, _addr)) => {
                    let control = Arc::clone(&control);
                    let stop = Arc::clone(&stop);
                    std::thread::Builder::new()
                        .name("frameiru-ipc-conn".into())
                        .spawn(move || handle_connection(stream, control, stop))
                        .expect("spawn ipc connection thread");
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(ACCEPT_POLL);
                }
                Err(e) => {
                    tracing::warn!("ipc accept error: {e}");
                    std::thread::sleep(ACCEPT_POLL);
                }
            }
        }
        self.unlink();
        Ok(())
    }

    fn unlink(&self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

impl Drop for IpcServer {
    fn drop(&mut self) {
        self.unlink();
    }
}

fn handle_connection(stream: UnixStream, control: Arc<dyn Control>, stop: Arc<AtomicBool>) {
    let _ = stream.set_read_timeout(Some(CONN_TIMEOUT));
    let mut reader = BufReader::new(stream);
    loop {
        let mut line = Vec::new();
        match reader.read_until(b'\n', &mut line) {
            Ok(0) => break, // peer closed
            Ok(_) => {}
            Err(_) => break,
        }
        let line = line.strip_suffix(b"\n").unwrap_or(&line);
        if line.is_empty() {
            continue;
        }

        let response = match decode::<IpcRequest>(line) {
            Ok(IpcRequest::SetBackground { mode }) => match control.set_background(mode) {
                Ok(()) => IpcResponse::Ok,
                Err(e) => IpcResponse::Error {
                    message: e.to_string(),
                },
            },
            Ok(IpcRequest::GetStatus) => IpcResponse::Status {
                status: control.status(),
            },
            Ok(IpcRequest::Stop) => {
                let result = control.stop();
                stop.store(true, Ordering::Relaxed);
                match result {
                    Ok(()) => IpcResponse::Ok,
                    Err(e) => IpcResponse::Error {
                        message: e.to_string(),
                    },
                }
            }
            Err(e) => IpcResponse::Error {
                message: e.to_string(),
            },
        };

        match encode(&response).and_then(|bytes| {
            reader.get_mut().write_all(&bytes)?;
            reader.get_mut().flush()?;
            Ok(())
        }) {
            Ok(()) => {}
            Err(e) => {
                tracing::warn!("ipc write failed: {e}");
                break;
            }
        }
        if stop.load(Ordering::Relaxed) {
            break;
        }
    }
}

/// Control-plane client; one persistent connection, serialized requests.
pub struct IpcClient {
    stream: std::sync::Mutex<UnixStream>,
    path: PathBuf,
}

impl IpcClient {
    pub fn connect(path: impl AsRef<Path>) -> Result<Self, FrameiruError> {
        let path = path.as_ref().to_path_buf();
        let stream = UnixStream::connect(&path)?;
        stream.set_read_timeout(Some(CONN_TIMEOUT))?;
        Ok(Self {
            stream: std::sync::Mutex::new(stream),
            path,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Sends one request and waits for its response.
    pub fn request(&self, request: &IpcRequest) -> Result<IpcResponse, FrameiruError> {
        let mut stream = self.stream.lock().expect("ipc client mutex poisoned");
        stream.write_all(&encode(request)?)?;
        stream.flush()?;

        let mut line = Vec::new();
        {
            let mut reader = BufReader::new(&mut *stream);
            reader.read_until(b'\n', &mut line)?;
        }
        let line = line.strip_suffix(b"\n").unwrap_or(&line);
        decode(line)
    }

    /// Convenience: change the background of the running pipeline.
    pub fn set_background(&self, mode: BackgroundMode) -> Result<(), FrameiruError> {
        match self.request(&IpcRequest::SetBackground { mode })? {
            IpcResponse::Ok => Ok(()),
            IpcResponse::Error { message } => {
                Err(FrameiruError::Internal(format!("ipc error: {message}")))
            }
            other => Err(FrameiruError::Internal(format!(
                "unexpected ipc response: {other:?}"
            ))),
        }
    }

    /// Convenience: fetch the runtime snapshot.
    pub fn status(&self) -> Result<StatusInfo, FrameiruError> {
        match self.request(&IpcRequest::GetStatus)? {
            IpcResponse::Status { status } => Ok(status),
            IpcResponse::Error { message } => {
                Err(FrameiruError::Internal(format!("ipc error: {message}")))
            }
            other => Err(FrameiruError::Internal(format!(
                "unexpected ipc response: {other:?}"
            ))),
        }
    }

    /// Convenience: stop the pipeline and shut the server down.
    pub fn stop(&self) -> Result<(), FrameiruError> {
        match self.request(&IpcRequest::Stop)? {
            IpcResponse::Ok => Ok(()),
            IpcResponse::Error { message } => {
                Err(FrameiruError::Internal(format!("ipc error: {message}")))
            }
            other => Err(FrameiruError::Internal(format!(
                "unexpected ipc response: {other:?}"
            ))),
        }
    }
}
