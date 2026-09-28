//! D-Bus service (feature `dbus`).
//!
//! Exports `io.frameiru.Control` on the session bus: `set_background` takes
//! the background mode as JSON, `get_status` returns the status snapshot as
//! JSON, and `stop` shuts the pipeline down. Serves forever (daemon role);
//! exits with an error when no session bus is reachable.

use std::sync::Arc;

use frameiru_core::error::FrameiruError;
use zbus::interface;

use crate::Control;

pub struct DbusProvider {
    control: Arc<dyn Control>,
}

#[interface(name = "io.frameiru.Control")]
impl DbusProvider {
    /// Changes the background; `json` is a serialized
    /// [`BackgroundMode`](frameiru_core::BackgroundMode).
    async fn set_background(&self, json: String) -> zbus::fdo::Result<()> {
        let mode: frameiru_core::BackgroundMode = serde_json::from_str(&json)
            .map_err(|e| zbus::fdo::Error::InvalidArgs(format!("bad background json: {e}")))?;
        self.control
            .set_background(mode)
            .map_err(|e| zbus::fdo::Error::Failed(e.to_string()))
    }

    /// Returns the current [`StatusInfo`] as JSON.
    async fn get_status(&self) -> zbus::fdo::Result<String> {
        serde_json::to_string(&self.control.status())
            .map_err(|e| zbus::fdo::Error::Failed(e.to_string()))
    }

    /// Stops the pipeline.
    async fn stop(&self) -> zbus::fdo::Result<()> {
        self.control
            .stop()
            .map_err(|e| zbus::fdo::Error::Failed(e.to_string()))
    }
}

/// Serves the D-Bus interface until the process exits (daemon role).
pub fn serve(control: Arc<dyn Control>) -> Result<(), FrameiruError> {
    zbus::block_on(async {
        let connection = zbus::connection::Connection::session()
            .await
            .map_err(|e| FrameiruError::Internal(format!("dbus session failed: {e}")))?;
        connection
            .object_server()
            .at("/io/frameiru/Control", DbusProvider { control })
            .await
            .map_err(|e| FrameiruError::Internal(format!("dbus export failed: {e}")))?;
        std::future::pending::<()>().await;
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::StatusInfo;

    /// `serve` must fail cleanly without a session bus (CI-safe).
    #[test]
    fn serve_fails_without_session_bus() {
        let control: Arc<dyn Control> = Arc::new(NoopControl);
        if std::env::var("DBUS_SESSION_BUS_ADDRESS").is_ok() {
            eprintln!("skipping: a session bus is present");
            return;
        }
        assert!(serve(control).is_err());
    }

    struct NoopControl;

    impl Control for NoopControl {
        fn set_background(
            &self,
            _mode: frameiru_core::BackgroundMode,
        ) -> Result<(), FrameiruError> {
            Ok(())
        }

        fn status(&self) -> StatusInfo {
            StatusInfo {
                running: false,
                resolution: None,
                capture_fps: 0.0,
                composite_fps: 0.0,
                frames_composited: 0,
                masks_computed: 0,
                latency_us: 0,
                background: frameiru_core::BackgroundMode::Passthrough,
            }
        }

        fn stop(&self) -> Result<(), FrameiruError> {
            Ok(())
        }
    }
}
