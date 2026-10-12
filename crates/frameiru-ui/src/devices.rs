//! Capture-device discovery for the camera selector dropdown: enumerates
//! V4L2 nodes, filters out loopback/metadata devices, and builds friendly
//! labels for the Slint combo box.

#[cfg(feature = "v4l2")]
use frameiru_webcam_utils::detect;

/// Represents an available video input device (mock source or physical V4L2 camera).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceEntry {
    /// Device identifier: `(mock)` or `/dev/videoN`.
    pub id: String,
    /// Friendly user-visible label: e.g. "Anker PowerConf C200 (/dev/video0)".
    pub display_name: String,
    /// True for supported webcams whose vendor controls are exposed in the drawer.
    pub is_anker_c200: bool,
}

/// Cleans raw V4L2 card/node name to remove truncated duplicate model strings.
/// E.g. "Anker PowerConf C200: Anker Pow" -> "Anker PowerConf C200".
#[cfg_attr(not(feature = "v4l2"), allow(dead_code))] // v4l2 discovery only
pub fn clean_device_name(name: &str) -> String {
    let trimmed = name.trim();
    if let Some((head, tail)) = trimmed.split_once(':') {
        let h = head.trim();
        let t = tail.trim();
        if !h.is_empty() && (h.starts_with(t) || t.starts_with(h)) {
            return if h.len() >= t.len() {
                h.to_string()
            } else {
                t.to_string()
            };
        }
    }
    trimmed.to_string()
}

/// Extracts trailing digits from a device path for natural numeric sorting.
#[cfg_attr(not(feature = "v4l2"), allow(dead_code))] // v4l2 discovery only
pub fn extract_trailing_number(path: &str) -> Option<u32> {
    path.rsplit(|c: char| !c.is_ascii_digit())
        .next()
        .and_then(|digits| digits.parse::<u32>().ok())
}

/// Enumerate available capture devices: filters out metadata-only and loopback devices.
pub fn detect_devices() -> Vec<DeviceEntry> {
    #[cfg_attr(
        not(feature = "v4l2"),
        allow(unused_mut) // v4l2 discovery pushes real devices into it
    )]
    let mut devices = vec![DeviceEntry {
        id: "(mock)".into(),
        display_name: "Mock Camera ((mock))".into(),
        is_anker_c200: false,
    }];

    #[cfg(feature = "v4l2")]
    {
        let mut real_devices = Vec::new();
        for node in frameiru_capture::v4l::context::enum_devices() {
            let path = node.path();
            // Verify device can be opened and actually supports VIDEO_CAPTURE
            let Ok(dev) = frameiru_capture::v4l::Device::with_path(path) else {
                continue;
            };
            let Ok(caps) = dev.query_caps() else {
                continue;
            };
            if !caps
                .capabilities
                .contains(frameiru_capture::v4l::capability::Flags::VIDEO_CAPTURE)
            {
                continue;
            }
            // Skip loopback devices (e.g. Frameiru virtual camera sink)
            if caps.driver.starts_with("v4l2 loopback") || caps.card.contains("Frameiru") {
                continue;
            }

            let raw_name = if !caps.card.trim().is_empty() {
                caps.card
            } else if let Some(name) = node.name() {
                name
            } else {
                "Camera".to_string()
            };

            let clean_name = clean_device_name(&raw_name);
            let display_name = format!("{clean_name} ({})", path.display());
            let is_anker_c200 = detect::is_anker_c200(path);
            real_devices.push((
                path.to_string_lossy().to_string(),
                display_name,
                is_anker_c200,
            ));
        }

        real_devices.sort_by(|a, b| {
            let num_a = extract_trailing_number(&a.0);
            let num_b = extract_trailing_number(&b.0);
            match (num_a, num_b) {
                (Some(na), Some(nb)) => na.cmp(&nb),
                _ => a.0.cmp(&b.0),
            }
        });

        for (id, display_name, is_anker_c200) in real_devices {
            devices.push(DeviceEntry {
                id,
                display_name,
                is_anker_c200,
            });
        }
    }

    devices
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cleans_device_names() {
        assert_eq!(
            clean_device_name("Anker PowerConf C200: Anker Pow"),
            "Anker PowerConf C200"
        );
        assert_eq!(
            clean_device_name("Integrated Camera: Integrated C"),
            "Integrated Camera"
        );
        assert_eq!(
            clean_device_name("Logitech Webcam C920"),
            "Logitech Webcam C920"
        );
        assert_eq!(clean_device_name(""), "");
    }

    #[test]
    fn extracts_trailing_number_correctly() {
        assert_eq!(extract_trailing_number("/dev/video0"), Some(0));
        assert_eq!(extract_trailing_number("/dev/video1"), Some(1));
        assert_eq!(extract_trailing_number("/dev/video10"), Some(10));
        assert_eq!(extract_trailing_number("(mock)"), None);
        assert_eq!(extract_trailing_number(""), None);
    }

    #[test]
    fn detects_devices_contains_mock_and_valid_capture_nodes() {
        let devices = detect_devices();
        assert!(!devices.is_empty());
        assert_eq!(devices[0].id, "(mock)");
        assert!(!devices[0].is_anker_c200, "mock source is never a C200");
        // Loopback devices and metadata-only nodes are excluded by
        // construction; anything real must look like a video node.
        for dev in &devices[1..] {
            assert!(dev.id.starts_with("/dev/video"));
            assert!(
                dev.display_name.contains(&dev.id),
                "display name must have device path hint"
            );
        }
    }
}
