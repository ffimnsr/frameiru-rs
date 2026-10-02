//! USB identity detection for webcams supported by the control drawer.
//!
//! The kernel exposes the USB `PRODUCT=<vid>/<pid>/<rev>` line (lower-case
//! hex) in the `uevent` file of each `videoN` device's sysfs entry.

use std::path::{Path, PathBuf};

/// Anker PowerConf C200 USB vendor id.
pub const ANKER_C200_VID: u16 = 0x291a;
/// Anker PowerConf C200 USB product id.
pub const ANKER_C200_PID: u16 = 0x3369;

/// True when `path` is a video node whose USB device is an Anker PowerConf C200.
pub fn is_anker_c200(path: impl AsRef<Path>) -> bool {
    usb_id(path).is_some_and(|(vid, pid)| vid == ANKER_C200_VID && pid == ANKER_C200_PID)
}

/// Reads the USB vendor/product id of the device behind a `/dev/videoN` node
/// from its sysfs `uevent` file.
pub fn usb_id(path: impl AsRef<Path>) -> Option<(u16, u16)> {
    let uevent = sysfs_uevent_path(path.as_ref())?;
    let content = std::fs::read_to_string(uevent).ok()?;
    parse_uevent(&content)
}

/// Maps a `/dev/videoN` node to `/sys/class/video4linux/videoN/device/uevent`.
fn sysfs_uevent_path(path: &Path) -> Option<PathBuf> {
    let name = path.file_name()?.to_str()?;
    if !name.starts_with("video") {
        return None;
    }
    Some(
        PathBuf::from("/sys/class/video4linux")
            .join(name)
            .join("device")
            .join("uevent"),
    )
}

/// Extracts `(vid, pid)` from the first `PRODUCT=` line of a uevent file.
pub fn parse_uevent(content: &str) -> Option<(u16, u16)> {
    content.lines().find_map(parse_product_line)
}

fn parse_product_line(line: &str) -> Option<(u16, u16)> {
    let value = line.trim().strip_prefix("PRODUCT=")?;
    let mut parts = value.split('/');
    let vid = u16::from_str_radix(parts.next()?, 16).ok()?;
    let pid = u16::from_str_radix(parts.next()?, 16).ok()?;
    Some((vid, pid))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_c200_product_line() {
        let content = "PRODUCT=291a/3369/8\nTYPE=0/0/0\n";
        assert_eq!(parse_uevent(content), Some((0x291a, 0x3369)));
    }

    #[test]
    fn parses_uppercase_hex() {
        assert_eq!(
            parse_uevent("PRODUCT=291A/3369/1\n"),
            Some((0x291a, 0x3369))
        );
    }

    #[test]
    fn rejects_lines_without_product() {
        assert_eq!(parse_uevent("DEVTYPE=usb_interface\n"), None);
        assert_eq!(parse_uevent("HID_PRODUCT=291a/3369\n"), None);
        assert_eq!(parse_uevent(""), None);
    }

    #[test]
    fn rejects_garbage_values() {
        assert_eq!(parse_uevent("PRODUCT=zzzz/3369/1\n"), None);
        assert_eq!(parse_uevent("PRODUCT=291a/\n"), None);
        assert_eq!(parse_uevent("PRODUCT=\n"), None);
        // A non-C200 camera must parse but not match.
        assert_eq!(
            parse_uevent("PRODUCT=046d/0825/1\n"),
            Some((0x046d, 0x0825))
        );
    }

    #[test]
    fn anker_id_matches_c200_only() {
        assert!(
            is_anker_c200_for_id(0x291a, 0x3369),
            "C200 USB id must match"
        );
        assert!(
            !is_anker_c200_for_id(0x291a, 0x1234),
            "same vendor, other product must not match"
        );
        assert!(
            !is_anker_c200_for_id(0x046d, 0x3369),
            "other vendor, same product must not match"
        );
    }

    fn is_anker_c200_for_id(vid: u16, pid: u16) -> bool {
        vid == ANKER_C200_VID && pid == ANKER_C200_PID
    }

    #[test]
    fn sysfs_path_is_derived_from_video_node_name() {
        assert_eq!(
            sysfs_uevent_path(Path::new("/dev/video3")),
            Some(PathBuf::from("/sys/class/video4linux/video3/device/uevent"))
        );
        assert_eq!(sysfs_uevent_path(Path::new("/dev/not-a-video")), None);
        assert_eq!(sysfs_uevent_path(Path::new("not-a-video")), None);
    }

    #[test]
    fn non_video_paths_are_never_anker() {
        assert!(!is_anker_c200("/dev/does-not-exist"));
        assert!(!is_anker_c200(""));
    }
}
