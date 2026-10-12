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
//! - Exclusivity is therefore detected per node from the announced caps
//!   themselves. The module-wide `exclusive_caps` sysfs array is not used:
//!   it always has `MAX_DEVICES` slots, unused slots read as `N` (the
//!   default), and its indices cannot be mapped to `/dev/videoN` reliably.

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

impl LoopbackNode {
    /// Whether this device announces its capabilities exclusively (the
    /// per-device `exclusive_caps` mode it was created with).
    ///
    /// Derived from the caps window the kernel exposes (see the module docs):
    /// an exclusive device announces OUTPUT while idle (or with only capture
    /// clients attached) and CAPTURE while a producer streams; a
    /// non-exclusive device always announces both. `None` when the node
    /// could not be opened for caps, i.e. the mode is unclassifiable.
    pub fn exclusive_caps(&self) -> Option<bool> {
        match (self.output_capable, self.capture_capable) {
            (true, false) | (false, true) => Some(true),
            (true, true) => Some(false),
            (false, false) => None,
        }
    }
}

/// Health of the virtual-camera sink, in probe priority order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoopbackStatus {
    /// A writable loopback node exists (or was opened); `exclusive_caps`
    /// reflects the device's caps window, which matters for Chrome/Zoom.
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
/// gone) via sysfs, so they still surface as classified failures instead of
/// silently vanishing.
fn is_loopback_sysfs(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
        return false;
    };
    is_loopback_sysfs_node(&Path::new("/sys/class/video4linux").join(name))
}

/// True when `dir` (a `/sys/class/video4linux/videoN` dir) is a v4l2loopback
/// node.
///
/// v4l2loopback registers its video devices without a parent device, so the
/// `device` symlink physical cameras have is absent; the reliable markers are
/// the per-device attributes the module creates (`state` on 0.13+, and
/// `format`/`buffers`/`max_openers` already on 0.12.x). The `device/driver`
/// link is still honored where some build happens to expose it.
fn is_loopback_sysfs_node(dir: &Path) -> bool {
    let driver = dir.join("device").join("driver");
    if std::fs::read_link(&driver)
        .map(|target| target.to_string_lossy().contains("v4l2loopback"))
        .unwrap_or(false)
    {
        return true;
    }
    ["state", "format", "buffers", "max_openers"]
        .iter()
        .any(|attr| dir.join(attr).is_file())
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

/// `exclusive_caps` mode of the loopback node currently at `path`; `None`
/// when no such loopback node is enumerable right now.
pub fn path_exclusive_caps(path: &Path) -> Option<bool> {
    loopback_nodes()
        .into_iter()
        .find(|node| node.path == path)?
        .exclusive_caps()
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

    let mut first_issue: Option<LoopbackStatus> = None;
    for idx in preferred_order(&nodes) {
        let node = &nodes[idx];
        let status = probe_node(node);
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

fn probe_node(node: &LoopbackNode) -> LoopbackStatus {
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
        // Output-capable node: the caps window decides exclusivity (idle
        // exclusive nodes are OUTPUT-only, non-exclusive nodes have both).
        Ok(_) => LoopbackStatus::Ready {
            path: node.path.clone(),
            exclusive_caps: node.exclusive_caps().unwrap_or(true),
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
    fn exclusive_caps_derived_from_the_caps_window() {
        // Regression: a single-device load (`exclusive_caps=1 video_nr=10`)
        // gives an idle OUTPUT-only node; it must read as exclusive even
        // though the module-wide sysfs array shows `Y` only in slot 0.
        let mut idle_exclusive = node("/dev/video10", "Frameiru");
        idle_exclusive.capture_capable = false;
        assert_eq!(idle_exclusive.exclusive_caps(), Some(true));

        // Same device while a producer streams: CAPTURE-only.
        let mut streaming_exclusive = node("/dev/video10", "Frameiru");
        streaming_exclusive.output_capable = false;
        assert_eq!(streaming_exclusive.exclusive_caps(), Some(true));

        // Non-exclusive devices always announce both caps.
        assert_eq!(
            node("/dev/video10", "Frameiru").exclusive_caps(),
            Some(false)
        );

        // Neither cap readable: unclassifiable.
        let mut unreadable = node("/dev/video10", "Frameiru");
        unreadable.capture_capable = false;
        unreadable.output_capable = false;
        assert_eq!(unreadable.exclusive_caps(), None);
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
            probe_node(&n),
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
            probe_node(&n),
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

    #[test]
    fn sysfs_markers_identify_loopback_nodes_without_a_device_link() {
        use std::sync::atomic::{AtomicU32, Ordering};
        static NEXT_TMP: AtomicU32 = AtomicU32::new(0);
        let n = NEXT_TMP.fetch_add(1, Ordering::Relaxed);
        let dir =
            std::env::temp_dir().join(format!("frameiru-loopback-test-{}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        // The v4l2loopback device layout (no `device` symlink at all).
        let node_linkless = dir.join("video10");
        std::fs::create_dir_all(&node_linkless).unwrap();
        std::fs::write(node_linkless.join("state"), "output\n").unwrap();
        std::fs::write(node_linkless.join("format"), "YUYV:640x480@30\n").unwrap();
        std::fs::write(node_linkless.join("buffers"), "2\n").unwrap();
        std::fs::write(node_linkless.join("max_openers"), "10\n").unwrap();
        assert!(is_loopback_sysfs_node(&node_linkless));
        // 0.12.x-style layout: no `state` yet.
        let node_012 = dir.join("video11");
        std::fs::create_dir_all(&node_012).unwrap();
        std::fs::write(node_012.join("format"), "YUYV:640x480@30\n").unwrap();
        assert!(is_loopback_sysfs_node(&node_012));
        // Physical camera with a driver link elsewhere: not loopback.
        let node_uvc = dir.join("video0");
        std::fs::create_dir_all(node_uvc.join("device")).unwrap();
        std::os::unix::fs::symlink(
            "/sys/bus/usb/drivers/uvcvideo",
            node_uvc.join("device").join("driver"),
        )
        .unwrap();
        assert!(!is_loopback_sysfs_node(&node_uvc));
        // A driver link to v4l2loopback still counts (older builds).
        let node_linked = dir.join("video12");
        std::fs::create_dir_all(node_linked.join("device")).unwrap();
        std::os::unix::fs::symlink(
            "/sys/bus/platform/drivers/v4l2loopback",
            node_linked.join("device").join("driver"),
        )
        .unwrap();
        assert!(is_loopback_sysfs_node(&node_linked));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
