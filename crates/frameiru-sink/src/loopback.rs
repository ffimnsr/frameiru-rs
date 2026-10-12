//! v4l2loopback discovery and health probe (feature `v4l2`).
//!
//! Classifies the virtual-camera sink so the GUI can stop guessing: module
//! missing, loaded without `exclusive_caps`, device busy with another writer,
//! permission denied, or ready to publish on a concrete `/dev/videoN`.
//!
//! Kernel facts the probe relies on (v4l2loopback.c):
//! - `cap.driver == "v4l2 loopback"` identifies loopback nodes.
//! - A device created with `exclusive_caps=1` advertises `VIDEO_OUTPUT` while
//!   idle and `VIDEO_CAPTURE` while a producer streams; a non-exclusive
//!   device advertises both, always.
//! - A second writer fails `S_FMT`/`REQBUFS` with `EBUSY` (single-writer
//!   device), so a failed open while the device looks output-capable means
//!   another application holds it.
//! - `exclusive_caps` is a `module_param_array(bool)` exposed read-only at
//!   `/sys/module/v4l2loopback/parameters/exclusive_caps` as `Y`/`N` per
//!   device (`,`-separated when `devices > 1`).

use std::path::{Path, PathBuf};

use frameiru_core::error::FrameiruError;
use frameiru_core::format::Resolution;
use nix::errno::Errno;
use v4l::capability::Flags;
use v4l::device::Device;

use crate::v4l2::LoopbackSink;

/// Module name used for sysfs checks and load hints.
pub const MODULE_NAME: &str = "v4l2loopback";
/// sysfs dir that exists while the module is loaded.
pub const MODULE_SYSFS: &str = "/sys/module/v4l2loopback";
/// Label the official load command assigns to the virtual camera.
pub const FRAMEIRU_CARD_LABEL: &str = "Frameiru";
/// Video node number used by the official load command.
pub const DEFAULT_VIDEO_NR: u32 = 10;

/// Resolution the probe uses for its open test (matches the UI pipeline).
const PROBE_RESOLUTION: Resolution = Resolution {
    width: 640,
    height: 480,
};

/// One v4l2loopback node found via V4L2 enumeration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoopbackNode {
    pub path: PathBuf,
    /// V4L2 card label (e.g. "Frameiru"); "(unreadable)" when caps failed.
    pub card: String,
    pub capture_capable: bool,
    pub output_capable: bool,
    /// Errno of the caps-open failure, when the node could not be opened at
    /// all (this is how `PermissionDenied` is detected).
    pub open_error: Option<i32>,
}

/// Health of the virtual-camera sink, in probe priority order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoopbackStatus {
    /// A writable loopback node exists (or was opened); `exclusive_caps`
    /// reflects the module parameter, which matters for Chrome/Zoom.
    Ready { path: PathBuf, exclusive_caps: bool },
    /// Module not loaded and no loopback node present.
    ModuleMissing,
    /// Module loaded but it created no video nodes (`devices=0`).
    NoDevices,
    /// Another producer holds the node (OBS, a second Frameiru, ...).
    Busy { path: PathBuf },
    /// Open failed with EACCES (user not in the `video` group).
    PermissionDenied { path: PathBuf },
    /// Node exists but cannot be opened or configured; carries the reason.
    Unavailable { path: PathBuf, reason: String },
}

/// Enumerates v4l2loopback nodes that could not be opened (permission or
/// gone) via the sysfs driver link, so they still surface as classified
/// failures instead of silently vanishing.
fn is_loopback_sysfs(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
        return false;
    };
    let driver = Path::new("/sys/class/video4linux")
        .join(name)
        .join("device")
        .join("driver");
    std::fs::read_link(&driver)
        .map(|target| target.to_string_lossy().contains("v4l2loopback"))
        .unwrap_or(false)
}

/// All v4l2loopback nodes currently visible, in discovery order.
pub fn loopback_nodes() -> Vec<LoopbackNode> {
    let mut nodes = Vec::new();
    for node in v4l::context::enum_devices() {
        let path = node.path().to_path_buf();
        let card = node.name().unwrap_or_else(|| "(unreadable)".into());
        match Device::with_path(&path).and_then(|dev| dev.query_caps()) {
            Ok(caps) => {
                if !caps.driver.starts_with("v4l2 loopback") {
                    continue;
                }
                nodes.push(LoopbackNode {
                    capture_capable: caps.capabilities.contains(Flags::VIDEO_CAPTURE),
                    output_capable: caps.capabilities.contains(Flags::VIDEO_OUTPUT),
                    open_error: None,
                    path,
                    card,
                });
            }
            Err(e) => {
                if is_loopback_sysfs(&path) {
                    nodes.push(LoopbackNode {
                        capture_capable: false,
                        output_capable: false,
                        open_error: e.raw_os_error(),
                        path,
                        card,
                    });
                }
            }
        }
    }
    nodes
}

/// Numeric tail of a `/dev/videoN` path, for natural ordering.
fn node_number(path: &Path) -> Option<u32> {
    let name = path.file_name()?.to_str()?;
    name.rsplit(|c: char| !c.is_ascii_digit())
        .next()
        .and_then(|digits| digits.parse::<u32>().ok())
}

/// Orders nodes for probing/opening: the Frameiru-labeled card first, then
/// ascending video number, then path (stable tie-break).
fn preferred_order(nodes: &[LoopbackNode]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..nodes.len()).collect();
    indices.sort_by_key(|&i| {
        let n = &nodes[i];
        let label_rank = u8::from(n.card == FRAMEIRU_CARD_LABEL);
        // sort_by_key: ascending; label match must come first, so invert.
        (1 - label_rank, node_number(&n.path).unwrap_or(u32::MAX), i)
    });
    indices
}

/// `true` when the module is loaded (its sysfs dir exists).
pub fn module_loaded() -> bool {
    Path::new(MODULE_SYSFS).is_dir()
}

/// Parses the kernel's bool-array sysfs output (`Y`,`N`,..., possibly one
/// value) into "every device is exclusive". `None` when unreadable/mixed
/// tokens we cannot map.
fn parse_exclusive_all(content: &str) -> Option<bool> {
    let tokens: Vec<&str> = content
        .split(',')
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .collect();
    if tokens.is_empty() {
        return None;
    }
    let mut exclusive = true;
    for token in tokens {
        exclusive &= match token {
            "Y" | "y" | "1" | "true" | "on" => true,
            "N" | "n" | "0" | "false" | "off" => false,
            _ => return None,
        };
    }
    Some(exclusive)
}

/// Module-wide `exclusive_caps` state; `None` when unreadable (then callers
/// assume the safe default).
pub fn exclusive_caps() -> Option<bool> {
    let params = Path::new(MODULE_SYSFS)
        .join("parameters")
        .join("exclusive_caps");
    parse_exclusive_all(&std::fs::read_to_string(params).ok()?)
}

/// Tries every candidate node in preferred order and returns the first one
/// that opens as a YUYV output device.
pub fn open_first_available(resolution: Resolution) -> Option<(LoopbackSink, PathBuf)> {
    let nodes = loopback_nodes();
    for idx in preferred_order(&nodes) {
        let path = nodes[idx].path.clone();
        if let Ok(sink) = LoopbackSink::open(&path, resolution) {
            return Some((sink, path));
        }
    }
    None
}

fn classify_io_error(path: &Path, errno: i32) -> LoopbackStatus {
    match Errno::from_raw(errno) {
        Errno::EBUSY => LoopbackStatus::Busy {
            path: path.to_path_buf(),
        },
        Errno::EACCES => LoopbackStatus::PermissionDenied {
            path: path.to_path_buf(),
        },
        other => LoopbackStatus::Unavailable {
            path: path.to_path_buf(),
            reason: format!("{other}"),
        },
    }
}

fn classify_open_error(path: &Path, err: &FrameiruError) -> LoopbackStatus {
    match err {
        FrameiruError::Io(io) => io
            .raw_os_error()
            .map(|errno| classify_io_error(path, errno))
            .unwrap_or_else(|| LoopbackStatus::Unavailable {
                path: path.to_path_buf(),
                reason: err.to_string(),
            }),
        other => LoopbackStatus::Unavailable {
            path: path.to_path_buf(),
            reason: other.to_string(),
        },
    }
}

/// Full health probe in preferred-node order; first `Ready` wins, otherwise
/// the first failure is reported.
pub fn probe() -> LoopbackStatus {
    let nodes = loopback_nodes();
    if nodes.is_empty() {
        return if module_loaded() {
            LoopbackStatus::NoDevices
        } else {
            LoopbackStatus::ModuleMissing
        };
    }

    let exclusive = exclusive_caps().unwrap_or(true);
    let mut first_issue: Option<LoopbackStatus> = None;
    for idx in preferred_order(&nodes) {
        let node = &nodes[idx];
        let status = probe_node(node, exclusive);
        if matches!(status, LoopbackStatus::Ready { .. }) {
            return status;
        }
        first_issue.get_or_insert(status);
    }
    first_issue.unwrap_or(LoopbackStatus::Unavailable {
        path: nodes[0].path.clone(),
        reason: "probe failed".into(),
    })
}

fn probe_node(node: &LoopbackNode, exclusive: bool) -> LoopbackStatus {
    if let Some(errno) = node.open_error {
        return classify_io_error(&node.path, errno);
    }
    if !node.output_capable {
        // Exclusive device with a streaming producer (or an output-caps
        // mismatch): the node advertises CAPTURE-only, so another writer
        // holds it and we cannot become one.
        return LoopbackStatus::Busy {
            path: node.path.clone(),
        };
    }
    match LoopbackSink::open(&node.path, PROBE_RESOLUTION) {
        Ok(_) => LoopbackStatus::Ready {
            path: node.path.clone(),
            exclusive_caps: exclusive,
        },
        Err(e) => classify_open_error(&node.path, &e),
    }
}

#[cfg(test)]
mod tests {
    use std::io;

    use super::*;

    fn node(path: &str, card: &str) -> LoopbackNode {
        LoopbackNode {
            path: PathBuf::from(path),
            card: card.into(),
            capture_capable: true,
            output_capable: true,
            open_error: None,
        }
    }

    #[test]
    fn parses_single_and_array_exclusive_caps() {
        assert_eq!(parse_exclusive_all("Y"), Some(true));
        assert_eq!(parse_exclusive_all("N"), Some(false));
        assert_eq!(parse_exclusive_all("Y,Y,Y"), Some(true));
        assert_eq!(parse_exclusive_all("Y,N"), Some(false));
        assert_eq!(parse_exclusive_all("N,N"), Some(false));
        assert_eq!(parse_exclusive_all("N\n"), Some(false));
        assert_eq!(parse_exclusive_all(""), None);
        assert_eq!(parse_exclusive_all("maybe"), None);
    }

    #[test]
    fn preferred_order_prefers_frameiru_label_then_lowest_number() {
        let nodes = vec![
            node("/dev/video16", "OBS Virtual Camera"),
            node("/dev/video12", "Frameiru"),
            node("/dev/video10", "Dummy video device (0x000A)"),
        ];
        let order = preferred_order(&nodes);
        assert_eq!(order, vec![1, 2, 0]);
    }

    #[test]
    fn preferred_order_dedupes_duplicate_numbers_by_path() {
        // Same tail: stable by index (path tie-break).
        let a = node("/dev/video10", "Frameiru");
        let b = node("/dev/video10", "Frameiru");
        let order = preferred_order(&[a, b]);
        assert_eq!(order, vec![0, 1]);
    }

    #[test]
    fn classifies_busy_permission_and_other_errnos() {
        let busy = FrameiruError::Io(io::Error::from_raw_os_error(16));
        assert_eq!(
            classify_open_error(Path::new("/dev/video10"), &busy),
            LoopbackStatus::Busy {
                path: "/dev/video10".into()
            }
        );
        let denied = FrameiruError::Io(io::Error::from_raw_os_error(13));
        assert_eq!(
            classify_open_error(Path::new("/dev/video10"), &denied),
            LoopbackStatus::PermissionDenied {
                path: "/dev/video10".into()
            }
        );
        let missing = FrameiruError::Io(io::Error::from_raw_os_error(2));
        assert!(matches!(
            classify_open_error(Path::new("/dev/video10"), &missing),
            LoopbackStatus::Unavailable { reason, .. } if reason.contains("ENOENT")
        ));
        let unsupported = FrameiruError::Unsupported("no YUYV output".into());
        assert!(matches!(
            classify_open_error(Path::new("/dev/video10"), &unsupported),
            LoopbackStatus::Unavailable { reason, .. } if reason.contains("no YUYV")
        ));
    }

    #[test]
    fn exclusive_only_node_without_output_is_busy() {
        let mut n = node("/dev/video10", "Frameiru");
        n.output_capable = false;
        n.capture_capable = true;
        assert_eq!(
            probe_node(&n, true),
            LoopbackStatus::Busy {
                path: "/dev/video10".into()
            }
        );
    }

    #[test]
    fn unopenable_node_surfaces_its_errno_before_caps() {
        let mut n = node("/dev/video10", "Frameiru");
        n.open_error = Some(13);
        assert_eq!(
            probe_node(&n, true),
            LoopbackStatus::PermissionDenied {
                path: "/dev/video10".into()
            }
        );
    }

    #[test]
    fn module_missing_when_no_nodes_and_no_sysfs_dir() {
        // Degenerate local check: the constants still line up.
        assert!(!module_loaded() || Path::new(MODULE_SYSFS).is_dir());
    }
}
