//! Anker PowerConf C200 webcam control drawer: reads/writes vendor UVC
//! controls and syncs the drawer widgets with the live device state.

use std::fs::File;

use frameiru_webcam_utils::controls::{
    find_control, format_bool, parse_bool, ControlInfo, ControlKind,
};
use frameiru_webcam_utils::{fov, v4l2, vendor, Error as WebcamError};

use crate::devices::DeviceEntry;
use crate::MainWindow;

/// Returns the statically registered control of `name` (always present).
pub fn webcam_control(name: &str) -> &'static ControlInfo {
    find_control(name).expect("webcam control registry is static")
}

/// Reads the current value of one control as display text.
fn read_webcam_value(fd: &File, info: &ControlInfo) -> Result<String, WebcamError> {
    match info.kind {
        ControlKind::VendorBool => Ok(format_bool(vendor::get_bool(fd, info.id as u8)?).into()),
        ControlKind::VendorU8 => Ok(vendor::get_u8(fd, info.id as u8)?.to_string()),
        ControlKind::VendorFov => Ok(fov::describe_value(fov::get(fd)?)),
        ControlKind::V4l2Bool => Ok(format_bool(v4l2::get(fd, info.id)? != 0).into()),
        ControlKind::V4l2Int | ControlKind::V4l2Menu => Ok(v4l2::get(fd, info.id)?.to_string()),
    }
}

/// Writes one control from its display text (mirrors the CLI `set` behavior).
pub(crate) fn write_webcam_value(
    fd: &File,
    info: &ControlInfo,
    value: &str,
) -> Result<(), WebcamError> {
    let invalid = || WebcamError::InvalidValue {
        name: info.name.to_owned(),
        value: value.to_owned(),
    };
    match info.kind {
        ControlKind::VendorBool => {
            let value = parse_bool(value).ok_or_else(invalid)?;
            vendor::set_bool(fd, info.id as u8, value)?;
        }
        ControlKind::VendorU8 => {
            let value = value.parse::<u8>().map_err(|_| invalid())?;
            vendor::set_u8(fd, info.id as u8, value)?;
        }
        ControlKind::VendorFov => {
            let value = fov::parse_value(value).ok_or_else(invalid)?;
            fov::set(fd, value)?;
        }
        ControlKind::V4l2Bool => {
            let value = parse_bool(value).ok_or_else(invalid)?;
            v4l2::set(fd, info.id, if value { 1 } else { 0 })?;
        }
        ControlKind::V4l2Int | ControlKind::V4l2Menu => {
            let value = value.parse::<i32>().map_err(|_| invalid())?;
            v4l2::set(fd, info.id, value)?;
        }
    }
    Ok(())
}

/// Syncs the drawer widget for one control with its live device value.
/// Failed reads leave the current widget state untouched.
pub(crate) fn push_webcam_control(window: &MainWindow, info: &ControlInfo, fd: &File) {
    let Ok(text) = read_webcam_value(fd, info) else {
        return;
    };
    match info.name {
        "fov" => {
            let raw = fov::get(fd).unwrap_or(0);
            let preset = fov::FOV_PRESETS
                .iter()
                .find(|(value, _, _)| *value == raw)
                .map(|(_, name, _)| *name)
                .unwrap_or("custom");
            window.set_webcam_fov(preset.into());
        }
        "hdr" => window.set_webcam_hdr(parse_bool(&text) == Some(true)),
        "horizontal_flip" => window.set_webcam_flip(parse_bool(&text) == Some(true)),
        "vertical_screen" => window.set_webcam_vertical(parse_bool(&text) == Some(true)),
        "anti_flicker" => window.set_webcam_anti_flicker(text != "0"),
        "brightness" => {
            if let Ok(value) = text.parse::<i32>() {
                window.set_webcam_brightness(value);
            }
        }
        _ => {}
    }
}

/// Read every drawer control from the selected webcam and sync the widgets.
pub fn refresh_webcam_state(window: &MainWindow, devices: &[DeviceEntry]) {
    window.set_webcam_error("".into());
    let index = window.get_selected_device() as usize;
    let Some(device) = devices.get(index).filter(|d| d.is_anker_c200) else {
        return;
    };
    let fd = match File::options().read(true).write(true).open(&device.id) {
        Ok(fd) => fd,
        Err(e) => {
            window
                .set_webcam_error(format!("webcam control: cannot open {}: {e}", device.id).into());
            return;
        }
    };
    for name in [
        "fov",
        "hdr",
        "horizontal_flip",
        "vertical_screen",
        "anti_flicker",
        "brightness",
    ] {
        push_webcam_control(window, webcam_control(name), &fd);
    }
    let brightness = webcam_control("brightness");
    if let Ok((min, max)) = v4l2::range(&fd, brightness.id) {
        window.set_webcam_brightness_min(min);
        window.set_webcam_brightness_max(max);
    }
}
