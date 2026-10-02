//! Control registry: names, kinds, and identifiers for every supported control.

/// How a control is read/written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlKind {
    /// Vendor extension unit boolean (1 byte, `0x01` / `0x00`).
    VendorBool,
    /// Vendor extension unit unsigned byte.
    VendorU8,
    /// Vendor extension unit FOV preset (u16 at offset 2, unit `0x06`, selector `0x10`).
    VendorFov,
    /// Standard V4L2 boolean control.
    V4l2Bool,
    /// Standard V4L2 integer control.
    V4l2Int,
    /// Standard V4L2 menu control.
    V4l2Menu,
}

/// Static description of one supported control.
#[derive(Debug, Clone, Copy)]
pub struct ControlInfo {
    pub name: &'static str,
    pub kind: ControlKind,
    /// Vendor extension-unit selector for vendor kinds, `V4L2_CID_*` for V4L2 kinds.
    pub id: u32,
    pub description: &'static str,
}

// Vendor selectors confirmed on the C200 (VID:PID 291a:3369), plus standard
// V4L2 control ids from `linux/videodev2.h`.
pub const CONTROLS: &[ControlInfo] = &[
    ControlInfo {
        name: "fov",
        kind: ControlKind::VendorFov,
        id: 0x10,
        description: "Vendor FOV preset/value",
    },
    ControlInfo {
        name: "hdr",
        kind: ControlKind::VendorBool,
        id: 0x13,
        description: "Vendor HDR toggle",
    },
    ControlInfo {
        name: "horizontal_flip",
        kind: ControlKind::VendorBool,
        id: 0x11,
        description: "Vendor horizontal mirror",
    },
    ControlInfo {
        name: "vertical_screen",
        kind: ControlKind::VendorBool,
        id: 0x0a,
        description: "Vendor vertical screen mode",
    },
    ControlInfo {
        name: "anti_flicker",
        kind: ControlKind::VendorU8,
        id: 0x12,
        description: "Vendor anti-flicker mode",
    },
    ControlInfo {
        name: "brightness",
        kind: ControlKind::V4l2Int,
        id: 0x0098_0900,
        description: "Standard UVC brightness",
    },
    ControlInfo {
        name: "contrast",
        kind: ControlKind::V4l2Int,
        id: 0x0098_0901,
        description: "Standard UVC contrast",
    },
    ControlInfo {
        name: "saturation",
        kind: ControlKind::V4l2Int,
        id: 0x0098_0902,
        description: "Standard UVC saturation",
    },
    ControlInfo {
        name: "hue",
        kind: ControlKind::V4l2Int,
        id: 0x0098_0903,
        description: "Standard UVC hue",
    },
    ControlInfo {
        name: "white_balance_automatic",
        kind: ControlKind::V4l2Bool,
        id: 0x0098_090c,
        description: "Standard UVC auto white balance",
    },
    ControlInfo {
        name: "gamma",
        kind: ControlKind::V4l2Int,
        id: 0x0098_0910,
        description: "Standard UVC gamma",
    },
    ControlInfo {
        name: "power_line_frequency",
        kind: ControlKind::V4l2Menu,
        id: 0x0098_0918,
        description: "Standard UVC power-line frequency",
    },
    ControlInfo {
        name: "white_balance_temperature",
        kind: ControlKind::V4l2Int,
        id: 0x0098_091a,
        description: "Standard UVC white balance temperature",
    },
    ControlInfo {
        name: "sharpness",
        kind: ControlKind::V4l2Int,
        id: 0x0098_091b,
        description: "Standard UVC sharpness",
    },
    ControlInfo {
        name: "auto_exposure",
        kind: ControlKind::V4l2Menu,
        id: 0x009a_0901,
        description: "Standard UVC auto exposure mode",
    },
    ControlInfo {
        name: "exposure_time_absolute",
        kind: ControlKind::V4l2Int,
        id: 0x009a_0902,
        description: "Standard UVC exposure time",
    },
    ControlInfo {
        name: "pan_absolute",
        kind: ControlKind::V4l2Int,
        id: 0x009a_0908,
        description: "Standard UVC pan",
    },
    ControlInfo {
        name: "tilt_absolute",
        kind: ControlKind::V4l2Int,
        id: 0x009a_0909,
        description: "Standard UVC tilt",
    },
    ControlInfo {
        name: "focus_absolute",
        kind: ControlKind::V4l2Int,
        id: 0x009a_090a,
        description: "Standard UVC focus",
    },
    ControlInfo {
        name: "focus_automatic_continuous",
        kind: ControlKind::V4l2Bool,
        id: 0x009a_090c,
        description: "Standard UVC autofocus",
    },
    ControlInfo {
        name: "zoom_absolute",
        kind: ControlKind::V4l2Int,
        id: 0x009a_090d,
        description: "Standard UVC zoom",
    },
];

/// Look up a control by name, case-insensitively (mirrors C `strcasecmp`).
pub fn find_control(name: &str) -> Option<&'static ControlInfo> {
    CONTROLS.iter().find(|c| c.name.eq_ignore_ascii_case(name))
}

/// Parse the accepted boolean tokens: `1|on|true|yes` / `0|off|false|no`.
pub fn parse_bool(text: &str) -> Option<bool> {
    match text.to_ascii_lowercase().as_str() {
        "1" | "on" | "true" | "yes" => Some(true),
        "0" | "off" | "false" | "no" => Some(false),
        _ => None,
    }
}

/// Canonical boolean rendering: `"on"` / `"off"`.
pub fn format_bool(value: bool) -> &'static str {
    if value {
        "on"
    } else {
        "off"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_control_is_case_insensitive() {
        assert_eq!(find_control("fov").unwrap().kind, ControlKind::VendorFov);
        assert_eq!(find_control("FOV").unwrap().kind, ControlKind::VendorFov);
        assert_eq!(
            find_control("Brightness").unwrap().kind,
            ControlKind::V4l2Int
        );
        assert_eq!(find_control("HDR").unwrap().kind, ControlKind::VendorBool);
    }

    #[test]
    fn find_control_unknown_returns_none() {
        assert!(find_control("banana").is_none());
        assert!(find_control("").is_none());
    }

    #[test]
    fn all_controls_have_unique_lowercase_names() {
        let mut names: Vec<&str> = CONTROLS.iter().map(|c| c.name).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), CONTROLS.len());
        assert!(CONTROLS
            .iter()
            .all(|c| c.name == c.name.to_ascii_lowercase()));
    }

    #[test]
    fn parse_bool_accepts_all_tokens() {
        assert_eq!(parse_bool("1"), Some(true));
        assert_eq!(parse_bool("on"), Some(true));
        assert_eq!(parse_bool("ON"), Some(true));
        assert_eq!(parse_bool("true"), Some(true));
        assert_eq!(parse_bool("yes"), Some(true));
        assert_eq!(parse_bool("0"), Some(false));
        assert_eq!(parse_bool("off"), Some(false));
        assert_eq!(parse_bool("false"), Some(false));
        assert_eq!(parse_bool("NO"), Some(false));
    }

    #[test]
    fn parse_bool_rejects_garbage() {
        assert_eq!(parse_bool("2"), None);
        assert_eq!(parse_bool("banana"), None);
        assert_eq!(parse_bool(" on"), None);
        assert_eq!(parse_bool(""), None);
    }

    #[test]
    fn format_bool_renders_on_off() {
        assert_eq!(format_bool(true), "on");
        assert_eq!(format_bool(false), "off");
    }
}
