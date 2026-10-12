//! KDE Plasma (KWin) minimize detection over the session D-Bus.
//!
//! Wayland exposes no client-side minimize signal, so on Plasma we load a
//! small KWin scripting script that matches our windows by PID and reports
//! every `minimizedChanged` back through a D-Bus method on our own object.
//! On any other desktop (no KWin scripting service) the watch stays inert
//! and `minimized()` simply reports `false`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use tokio::runtime::Runtime;
use tokio::sync::mpsc;
use zbus::interface;

/// Own D-Bus name/object the KWin script reports into.
const DBUS_NAME: &str = "org.frameiru.Pip";
const DBUS_PATH: &str = "/org/frameiru/Pip";
const DBUS_IFACE: &str = "org.frameiru.Pip";
/// KWin scripting service.
const KWIN_SERVICE: &str = "org.kde.KWin";
const KWIN_PATH: &str = "/Scripting";
const KWIN_IFACE: &str = "org.kde.KWin.Scripting";

/// Receives `emit-minimized(bool)` calls from the KWin script. The method
/// handler runs on the zbus executor thread; the flag is atomic so the UI
/// thread can read it from the poll tick.
struct PipSink {
    minimized: Arc<AtomicBool>,
}

#[interface(name = "org.frameiru.Pip")]
impl PipSink {
    async fn emit_minimized(&self, minimized: bool) {
        self.minimized.store(minimized, Ordering::Relaxed);
    }
}

/// Generates the KWin script for `pid` (our process). Pure, so the
/// interpolation is unit-testable.
fn kwin_script(pid: u32) -> String {
    format!(
        r#"// frameiru PiP minimize watch
const SB = "{DBUS_NAME}";
const PB = "{DBUS_PATH}";
const IB = "{DBUS_IFACE}";
function track(w) {{
    try {{
        callDBus(SB, PB, IB, "emitMinimized", w.minimized);
        w.minimizedChanged.connect(() => callDBus(SB, PB, IB, "emitMinimized", w.minimized));
    }} catch (e) {{}}
}}
workspace.windowList().filter(w => w.pid === {pid}).forEach(track);
workspace.windowAdded.connect(w => {{ if (w.pid === {pid}) track(w); }});
"#
    )
}

enum Cmd {
    Enable,
    Disable,
}

/// Loads/unloads the watch script and keeps the D-Bus connection alive.
async fn worker(mut rx: mpsc::Receiver<Cmd>, minimized: Arc<AtomicBool>, pid: u32) {
    let mut conn: Option<zbus::Connection> = None;
    let mut script_id: Option<u32> = None;
    while let Some(cmd) = rx.recv().await {
        match cmd {
            Cmd::Enable => {
                if conn.is_none() {
                    // Session bus + own D-Bus name so the script can reach us.
                    let Ok(c) = zbus::Connection::session().await else {
                        continue;
                    };
                    let Ok(ok) = c
                        .object_server()
                        .at(
                            DBUS_PATH,
                            PipSink {
                                minimized: minimized.clone(),
                            },
                        )
                        .await
                    else {
                        continue;
                    };
                    if !ok {
                        // Object already served (duplicate enable).
                        continue;
                    }
                    let Ok(reply) = c
                        .request_name_with_flags(
                            DBUS_NAME,
                            zbus::fdo::RequestNameFlags::DoNotQueue.into(),
                        )
                        .await
                    else {
                        continue;
                    };
                    if !matches!(
                        reply,
                        zbus::fdo::RequestNameReply::PrimaryOwner
                            | zbus::fdo::RequestNameReply::AlreadyOwner
                    ) {
                        // Another frameiru instance owns the name.
                        continue;
                    }
                    conn = Some(c);
                }
                let Some(c) = conn.as_ref() else {
                    continue;
                };
                if script_id.is_none() {
                    let Ok(proxy) = zbus::Proxy::new(c, KWIN_SERVICE, KWIN_PATH, KWIN_IFACE).await
                    else {
                        continue;
                    };
                    let Ok(id) = proxy
                        .call_method("loadScript", &(kwin_script(pid)))
                        .await
                        .and_then(|reply| reply.body().deserialize())
                    else {
                        // No KWin scripting service (non-Plasma desktop).
                        continue;
                    };
                    let _ = proxy.call_method("runScript", &(id)).await;
                    script_id = Some(id);
                }
            }
            Cmd::Disable => {
                if let (Some(c), Some(id)) = (conn.as_ref(), script_id.take()) {
                    let Ok(proxy) = zbus::Proxy::new(c, KWIN_SERVICE, KWIN_PATH, KWIN_IFACE).await
                    else {
                        continue;
                    };
                    let _ = proxy.call_method("unloadScript", &(id)).await;
                }
            }
        }
    }
}

/// Controller-side handle: owns the tokio runtime and the worker task.
/// Created once; `set_enabled` drives the worker over a channel. The
/// minimize flag is shared with the poll loop (via [`MiniPreviewController`]).
pub struct KWinWatch {
    rt: Runtime,
    tx: mpsc::Sender<Cmd>,
}

impl KWinWatch {
    pub fn new(pid: u32, minimized: Arc<AtomicBool>) -> Self {
        let rt = Runtime::new().expect("tokio runtime for the KWin watch");
        let (tx, rx) = mpsc::channel(4);
        let flag = Arc::clone(&minimized);
        rt.spawn(async move { worker(rx, flag, pid).await });
        Self { rt, tx }
    }

    /// Enable/disable the KWin-scripted watch. Idempotent; D-Bus work stays
    /// on the worker task, so this never blocks the UI thread.
    pub fn set_enabled(&self, enabled: bool) {
        let _ = self.rt.block_on(
            self.tx
                .send(if enabled { Cmd::Enable } else { Cmd::Disable }),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn script_contains_pid_and_watch_connection() {
        let script = kwin_script(4242);
        assert!(script.contains("w.pid === 4242"));
        assert!(script.contains("minimizedChanged.connect"));
        assert!(script.contains("workspace.windowAdded.connect"));
        assert!(script.contains(DBUS_NAME));
        assert!(script.contains(DBUS_PATH));
        assert!(script.contains("emitMinimized"));
    }

    #[test]
    fn script_escapes_braces() {
        // The `{{`/`}}` in the JS template must have become single braces.
        assert!(!kwin_script(1).contains("{{"));
        assert!(!kwin_script(1).contains("}}"));
    }
}
