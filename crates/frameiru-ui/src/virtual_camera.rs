//! Virtual camera (v4l2loopback sink) status and privileged loader for the
//! Slint GUI (layer 1 + 2 of the sink UX: honest status, one-click enable).
//!
//! [`SinkState::probe`] classifies the loopback sink without guessing; the
//! view mapping turns that into banner/pill text; the helper binary (see
//! `frameiru-loopback-helper`) performs the privileged `modprobe` through
//! `pkexec` so the user never touches a terminal.

use std::path::{Path, PathBuf};
use std::process::Command;

use slint::SharedString;

use crate::MainWindow;

/// Name of the privileged helper binary the UI locates and pkexec runs.
const HELPER_BIN: &str = "frameiru-loopback-helper";

/// Fallback shown (and copy-pasteable) when no helper binary is installed.
pub const MODPROBE_HINT: &str =
    "sudo modprobe v4l2loopback exclusive_caps=1 card_label=\"Frameiru\" video_nr=10";

/// UI-facing health of the virtual camera. Mirrors the probe classification
/// but adds `NoSupport` for builds without the `v4l2` feature (in such builds
/// only `NoSupport` is ever constructed).
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(not(feature = "v4l2"), allow(dead_code))]
pub enum SinkState {
    /// A writable loopback node exists; `exclusive` reflects the device's
    /// announced caps (its `exclusive_caps` mode).
    Ready { path: PathBuf, exclusive: bool },
    /// Kernel module not loaded.
    ModuleMissing,
    /// Module loaded but created no video nodes (`devices=0`).
    NoDevices,
    /// Another producer holds the node.
    Busy { path: PathBuf },
    /// Open failed EACCES (user not in `video` group).
    Denied { path: PathBuf },
    /// Node present but unusable; carries the reason.
    Failed {
        path: Option<PathBuf>,
        reason: String,
    },
    /// Built without `v4l2` support (only constructed in such builds, hence
    /// dead code analysis sees it unused here).
    #[allow(dead_code)]
    NoSupport,
}

impl SinkState {
    /// Live probe of the virtual camera sink.
    pub fn probe() -> Self {
        #[cfg(feature = "v4l2")]
        {
            frameiru_sink::loopback::probe().into()
        }
        #[cfg(not(feature = "v4l2"))]
        {
            SinkState::NoSupport
        }
    }

    /// Whether running the pkexec helper (`loopback-load`) can fix this state.
    pub fn needs_helper(&self) -> bool {
        matches!(
            self,
            SinkState::ModuleMissing
                | SinkState::NoDevices
                | SinkState::Ready {
                    exclusive: false,
                    ..
                }
        )
    }

    /// Whether a virtual camera is publishable right now (test-facing
    /// convenience; the app uses the banner view instead).
    #[cfg(test)]
    pub fn is_ready(&self) -> bool {
        matches!(self, SinkState::Ready { .. })
    }
}

#[cfg(feature = "v4l2")]
impl From<frameiru_sink::LoopbackStatus> for SinkState {
    fn from(status: frameiru_sink::LoopbackStatus) -> Self {
        use frameiru_sink::LoopbackStatus as L;
        match status {
            L::Ready {
                path,
                exclusive_caps,
            } => SinkState::Ready {
                path,
                exclusive: exclusive_caps,
            },
            L::ModuleMissing => SinkState::ModuleMissing,
            L::NoDevices => SinkState::NoDevices,
            L::Busy { path } => SinkState::Busy { path },
            L::PermissionDenied { path } => SinkState::Denied { path },
            L::Unavailable { path, reason } => SinkState::Failed {
                path: Some(path),
                reason,
            },
        }
    }
}

/// Everything the window needs to render the sink status strip.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SinkView {
    /// Green pill: publishing to a virtual camera.
    pub ok: bool,
    /// Amber pill: publishing, but Chrome/Zoom may not see the camera.
    pub warn: bool,
    /// Pill value, e.g. "/dev/video10", "OFF", "BUSY".
    pub pill: String,
    /// Banner body; empty hides the banner.
    pub message: String,
    /// Whether the banner shows an action button.
    pub action_visible: bool,
    /// Button label ("LOAD VIRTUAL CAMERA", "RELOAD VIRTUAL CAMERA", "RETRY").
    pub action_text: String,
}

/// Maps a sink state to user-facing text. `status_message` overrides the
/// banner body (used for "waiting for authorization…" and helper failures).
pub fn view_for(state: &SinkState, status_message: &str) -> SinkView {
    let (ok, warn, pill, message, action_visible, action_text) = match state {
        SinkState::Ready {
            path,
            exclusive: true,
        } => (
            true,
            false,
            path.display().to_string(),
            String::new(),
            false,
            String::new(),
        ),
        SinkState::Ready {
            path,
            exclusive: false,
        } => (
            true,
            true,
            path.display().to_string(),
            format!(
                "Virtual camera {} is in non-exclusive mode — Chrome and Zoom \
                 may not list it. Reload the module with exclusive_caps=1.",
                path.display()
            ),
            true,
            "RELOAD VIRTUAL CAMERA".into(),
        ),
        SinkState::ModuleMissing => (
            false,
            false,
            "OFF".into(),
            "Virtual camera not available: the v4l2loopback kernel module is \
             not loaded. Zoom/Meet/Chrome cannot see Frameiru."
                .into(),
            true,
            "LOAD VIRTUAL CAMERA".into(),
        ),
        SinkState::NoDevices => (
            false,
            false,
            "OFF".into(),
            "v4l2loopback is loaded but created no video nodes (devices=0); \
             reload it for Frameiru."
                .into(),
            true,
            "LOAD VIRTUAL CAMERA".into(),
        ),
        SinkState::Busy { path } => (
            false,
            false,
            "BUSY".into(),
            format!(
                "Virtual camera {} is in use by another application (e.g. \
                 OBS). Close it, then retry.",
                path.display()
            ),
            true,
            "RETRY".into(),
        ),
        SinkState::Denied { path } => (
            false,
            false,
            "DENIED".into(),
            format!(
                "Permission denied opening {} — is your user in the `video` \
                 group? (sudo usermod -aG video $USER)",
                path.display()
            ),
            true,
            "RETRY".into(),
        ),
        SinkState::Failed { path, reason } => (
            false,
            false,
            "FAULT".into(),
            match path {
                Some(path) => format!("Cannot use {}: {reason}", path.display()),
                None => reason.clone(),
            },
            true,
            "RETRY".into(),
        ),
        SinkState::NoSupport => (
            false,
            false,
            "N/A".into(),
            "Built without v4l2 support — rebuild frameiru-ui with the v4l2 \
             feature."
                .into(),
            false,
            String::new(),
        ),
    };

    SinkView {
        ok,
        warn,
        pill,
        message: if status_message.is_empty() {
            message
        } else {
            status_message.to_string()
        },
        action_visible,
        action_text,
    }
}

/// Pushes a [`SinkView`] into the window properties.
pub fn apply(window: &MainWindow, view: &SinkView) {
    window.set_sink_pill(SharedString::from(view.pill.clone()));
    window.set_sink_ok(view.ok);
    window.set_sink_warn(view.warn);
    window.set_sink_message(SharedString::from(view.message.clone()));
    window.set_sink_action_visible(view.action_visible);
    window.set_sink_action_text(SharedString::from(view.action_text.clone()));
}

/// Runs the sink-state view pipeline against the live probe and applies it.
pub fn refresh(window: &MainWindow) -> SinkState {
    let state = SinkState::probe();
    apply(window, &view_for(&state, ""));
    state
}

/// One sink-watch observation, reported back to the app for UI updates.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(not(feature = "v4l2"), allow(dead_code))]
pub enum WatchEvent {
    /// A loopback device was opened and hot-swapped into the running engine.
    Attached { path: PathBuf },
    /// The device set changed but nothing could be attached; the probe state
    /// (Busy, Denied, ...) is the current truth.
    Probe(SinkState),
}

/// Polls the loopback device set while the engine runs on a mock sink and
/// hot-attaches a real sink the moment a usable device appears (layer 5:
/// self-healing — loading the module externally, or another app releasing
/// the device, takes effect without an engine restart).
///
/// Cheap by design: each tick is a sysfs/V4L enumeration; the open attempt
/// only happens when the device set changed or a previous attempt failed
/// long enough ago to count as a retry.
#[cfg(feature = "v4l2")]
pub struct SinkWatcher {
    known_nodes: Vec<PathBuf>,
    last_failure: Option<std::time::Instant>,
    timer: slint::Timer,
}

/// Retry interval after a failed attach (device busy with another app).
#[cfg(feature = "v4l2")]
const SINK_WATCH_RETRY: std::time::Duration = std::time::Duration::from_secs(5);

#[cfg(feature = "v4l2")]
impl SinkWatcher {
    pub fn new() -> Self {
        Self {
            known_nodes: Vec::new(),
            last_failure: None,
            timer: slint::Timer::default(),
        }
    }

    /// Starts the periodic UI-thread poll (callback may capture `Rc`-state:
    /// the timer never leaves the UI thread).
    pub fn start(&self, every: std::time::Duration, callback: impl FnMut() + 'static) {
        self.timer
            .start(slint::TimerMode::Repeated, every, callback);
    }

    /// One tick: enumerates loopback nodes and, when the set changed (or a
    /// retry is due), tries to attach the first usable device.
    pub fn tick(
        &mut self,
        handle: &frameiru_pipeline::PipelineHandle,
        resolution: frameiru_core::format::Resolution,
    ) -> Option<WatchEvent> {
        let nodes = frameiru_sink::loopback::loopback_nodes();
        let paths: Vec<PathBuf> = nodes.iter().map(|n| n.path.clone()).collect();
        let set_changed = paths != self.known_nodes;
        self.known_nodes = paths;
        let retry_due = self
            .last_failure
            .is_some_and(|t| t.elapsed() >= SINK_WATCH_RETRY);
        if !set_changed && !retry_due {
            return None;
        }

        match frameiru_sink::loopback::open_first_available(resolution) {
            Some((sink, path)) => {
                self.last_failure = None;
                match handle.swap_sink(Box::new(sink)) {
                    Ok(()) => Some(WatchEvent::Attached { path }),
                    Err(e) => {
                        // Pipeline shutting down under us: report nothing.
                        eprintln!("sink swap rejected: {e}");
                        None
                    }
                }
            }
            None => {
                self.last_failure = Some(std::time::Instant::now());
                // Only a fresh device set deserves a full probe (Busy/
                // Denied/... classification); silent retries keep the UI calm.
                if set_changed {
                    Some(WatchEvent::Probe(SinkState::probe()))
                } else {
                    None
                }
            }
        }
    }
}

/// Builds without `v4l2`: the engine never publishes, so the watcher is a
/// no-op placeholder (keeps `app.rs` free of feature gates).
#[cfg(not(feature = "v4l2"))]
pub struct SinkWatcher;

#[cfg(not(feature = "v4l2"))]
impl SinkWatcher {
    pub fn new() -> Self {
        Self
    }

    pub fn start(&self, _every: std::time::Duration, _callback: impl FnMut() + 'static) {}

    pub fn tick(
        &mut self,
        _handle: &frameiru_pipeline::PipelineHandle,
        _resolution: frameiru_core::format::Resolution,
    ) -> Option<WatchEvent> {
        None
    }
}

impl Default for SinkWatcher {
    fn default() -> Self {
        Self::new()
    }
}

/// First existing candidate; `None` when none of them exists.
fn first_existing(candidates: &[PathBuf]) -> Option<PathBuf> {
    candidates.iter().find(|p| p.is_file()).cloned()
}

/// Locates the privileged helper: env override, system installs, then the
/// directory of the running UI binary (dev builds).
pub fn find_helper() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("FRAMEIRU_LOOPBACK_HELPER") {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Some(path);
        }
    }
    let mut candidates = vec![
        PathBuf::from(format!("/usr/libexec/{HELPER_BIN}")),
        PathBuf::from(format!("/usr/local/libexec/{HELPER_BIN}")),
    ];
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            candidates.push(dir.join(HELPER_BIN));
        }
    }
    first_existing(&candidates)
}

/// Outcome of one privileged load attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HelperOutcome {
    pub ok: bool,
    /// Combined helper output (already includes the reason on failure).
    pub output: String,
}

/// Runs the helper once via `pkexec`. `helper == None` reports the manual
/// fallback command instead of pretending.
pub fn run_helper(helper: Option<&Path>) -> HelperOutcome {
    let Some(helper) = helper else {
        return HelperOutcome {
            ok: false,
            output: format!(
                "Privileged helper not installed. Install packaging files \
                 (see packaging/README.md) or run manually:\n{MODPROBE_HINT}"
            ),
        };
    };

    let output = match Command::new("pkexec")
        .arg(helper)
        .arg("loopback-load")
        .output()
    {
        Ok(output) => output,
        Err(e) => {
            return HelperOutcome {
                ok: false,
                output: format!("cannot start pkexec: {e}"),
            };
        }
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let mut detail = format!("{stdout}{stderr}").trim().to_string();
    if let Some(code) = output.status.code() {
        // pkexec: 126 = authentication dialog dismissed, 127 = not authorized.
        let suffix = match code {
            126 => " (authentication cancelled)",
            127 => " (not authorized)",
            _ => "",
        };
        if !detail.is_empty() {
            detail.push_str(suffix);
        } else if !suffix.is_empty() {
            detail = suffix.trim_start_matches(' ').to_string();
        }
    }
    if detail.is_empty() {
        detail = format!("helper exited with {}", output.status);
    }

    HelperOutcome {
        ok: output.status.success(),
        output: detail,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ready_exclusive_shows_no_banner() {
        let state = SinkState::Ready {
            path: "/dev/video10".into(),
            exclusive: true,
        };
        let view = view_for(&state, "");
        assert!(view.ok);
        assert!(!view.warn);
        assert_eq!(view.pill, "/dev/video10");
        assert!(view.message.is_empty());
        assert!(!view.action_visible);
    }

    #[test]
    fn non_exclusive_ready_warns_and_offers_reload() {
        let state = SinkState::Ready {
            path: "/dev/video10".into(),
            exclusive: false,
        };
        let view = view_for(&state, "");
        assert!(view.ok);
        assert!(view.warn);
        assert!(view.action_visible);
        assert_eq!(view.action_text, "RELOAD VIRTUAL CAMERA");
        assert!(state.needs_helper());
    }

    #[test]
    fn missing_module_offers_load() {
        let view = view_for(&SinkState::ModuleMissing, "");
        assert!(!view.ok);
        assert_eq!(view.pill, "OFF");
        assert_eq!(view.action_text, "LOAD VIRTUAL CAMERA");
        assert!(SinkState::ModuleMissing.needs_helper());
        assert!(!SinkState::ModuleMissing.is_ready());
    }

    #[test]
    fn busy_and_denied_only_retry() {
        for state in [
            SinkState::Busy {
                path: "/dev/video10".into(),
            },
            SinkState::Denied {
                path: "/dev/video10".into(),
            },
            SinkState::Failed {
                path: Some("/dev/video10".into()),
                reason: "broken".into(),
            },
        ] {
            let view = view_for(&state, "");
            assert!(!view.ok);
            assert!(!state.needs_helper());
            assert_eq!(view.action_text, "RETRY");
        }
    }

    #[test]
    fn status_message_overrides_banner_body() {
        let view = view_for(&SinkState::ModuleMissing, "waiting for authorization…");
        assert_eq!(view.message, "waiting for authorization…");
        assert_eq!(view.pill, "OFF");
    }

    #[cfg(feature = "v4l2")]
    #[test]
    fn maps_sink_probe_statuses() {
        use frameiru_sink::LoopbackStatus;
        let cases = [
            (
                LoopbackStatus::Ready {
                    path: "/dev/video10".into(),
                    exclusive_caps: true,
                },
                SinkState::Ready {
                    path: "/dev/video10".into(),
                    exclusive: true,
                },
            ),
            (
                LoopbackStatus::Ready {
                    path: "/dev/video10".into(),
                    exclusive_caps: false,
                },
                SinkState::Ready {
                    path: "/dev/video10".into(),
                    exclusive: false,
                },
            ),
            (LoopbackStatus::ModuleMissing, SinkState::ModuleMissing),
            (LoopbackStatus::NoDevices, SinkState::NoDevices),
            (
                LoopbackStatus::Busy {
                    path: "/dev/video10".into(),
                },
                SinkState::Busy {
                    path: "/dev/video10".into(),
                },
            ),
            (
                LoopbackStatus::PermissionDenied {
                    path: "/dev/video10".into(),
                },
                SinkState::Denied {
                    path: "/dev/video10".into(),
                },
            ),
            (
                LoopbackStatus::Unavailable {
                    path: "/dev/video10".into(),
                    reason: "boom".into(),
                },
                SinkState::Failed {
                    path: Some("/dev/video10".into()),
                    reason: "boom".into(),
                },
            ),
        ];
        for (probe, expected) in cases {
            assert_eq!(SinkState::from(probe), expected);
        }
    }

    #[test]
    fn helper_hint_when_not_installed() {
        let outcome = run_helper(None);
        assert!(!outcome.ok);
        assert!(outcome
            .output
            .contains("modprobe v4l2loopback exclusive_caps=1"));
    }
}
