//! `frameiru-webcam-utils` binary: read/set Anker PowerConf C200 controls.
//!
//! Behavior mirrors the C reference tool `erans/anker-powerconf-c200-linux-tools`:
//! same commands, output formatting, and warning-on-readback-failure semantics.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::exit;

use anyhow::Context;
use clap::{Parser, Subcommand};
use frameiru_webcam_utils::controls::{
    find_control, format_bool, parse_bool, ControlInfo, ControlKind, CONTROLS,
};
use frameiru_webcam_utils::{fov, v4l2, vendor, Error};

#[derive(Debug, Parser)]
#[command(
    name = "frameiru-webcam-utils",
    version,
    about = "Control an Anker PowerConf C200 webcam (vendor UVC extension unit and standard V4L2 controls)"
)]
struct Cli {
    /// Video device node (extension-unit queries go through this node)
    #[arg(long, default_value = "/dev/video0")]
    device: PathBuf,

    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// List all supported controls
    ListControls,
    /// Read a control value
    Get { control: String },
    /// Set a control value, then print the resulting value
    Set { control: String, value: String },
    /// Compatibility alias for `get fov`
    #[command(name = "get-fov")]
    GetFov,
    /// Compatibility alias for `set fov`
    #[command(name = "set-fov")]
    SetFov { value: String },
}

fn open_device(path: &Path) -> Result<File, Error> {
    File::options()
        .read(true)
        .write(true)
        .open(path)
        .map_err(|source| Error::OpenFailed {
            path: path.to_path_buf(),
            source,
        })
}

fn resolve_control(name: &str) -> Result<&'static ControlInfo, Error> {
    find_control(name).ok_or_else(|| Error::UnknownControl {
        name: name.to_owned(),
    })
}

/// Print the current value of a control (readback after `set` too).
fn print_value(fd: &File, info: &ControlInfo) -> Result<(), Error> {
    match info.kind {
        ControlKind::VendorBool => {
            println!("{}", format_bool(vendor::get_bool(fd, info.id as u8)?));
        }
        ControlKind::VendorU8 => {
            println!("{}", vendor::get_u8(fd, info.id as u8)?);
        }
        ControlKind::VendorFov => {
            println!("{}", fov::describe_value(fov::get(fd)?));
        }
        ControlKind::V4l2Bool => {
            println!("{}", format_bool(v4l2::get(fd, info.id)? != 0));
        }
        ControlKind::V4l2Int | ControlKind::V4l2Menu => {
            println!("{}", v4l2::get(fd, info.id)?);
        }
    }
    Ok(())
}

fn parse_u8(text: &str) -> Option<u8> {
    text.parse().ok()
}

fn parse_i32(text: &str) -> Option<i32> {
    text.parse().ok()
}

fn set_value(fd: &File, info: &ControlInfo, text: &str) -> Result<(), Error> {
    let invalid = || Error::InvalidValue {
        name: info.name.to_owned(),
        value: text.to_owned(),
    };
    match info.kind {
        ControlKind::VendorBool => {
            let value = parse_bool(text).ok_or_else(invalid)?;
            vendor::set_bool(fd, info.id as u8, value)?;
        }
        ControlKind::VendorU8 => {
            let value = parse_u8(text).ok_or_else(invalid)?;
            vendor::set_u8(fd, info.id as u8, value)?;
        }
        ControlKind::VendorFov => {
            let value = fov::parse_value(text).ok_or_else(invalid)?;
            fov::set(fd, value)?;
        }
        ControlKind::V4l2Bool => {
            let value = parse_bool(text).ok_or_else(invalid)?;
            v4l2::set(fd, info.id, if value { 1 } else { 0 })?;
        }
        ControlKind::V4l2Int | ControlKind::V4l2Menu => {
            let value = parse_i32(text).ok_or_else(invalid)?;
            v4l2::set(fd, info.id, value)?;
        }
    }
    Ok(())
}

/// Mirrors C `open_device` + `handle_get_control` and its error message.
fn handle_get(device: &Path, info: &'static ControlInfo) -> anyhow::Result<()> {
    let fd = open_device(device)?;
    print_value(&fd, info)
        .with_context(|| format!("failed to read {} from {}", info.name, device.display()))
}

/// Mirrors C `handle_set_control`; a failed readback prints a warning and
/// returns exit code 1, matching the reference tool.
fn handle_set(device: &Path, info: &'static ControlInfo, value: &str) -> anyhow::Result<i32> {
    let fd = open_device(device)?;
    set_value(&fd, info, value)
        .with_context(|| format!("failed to set {} on {}", info.name, device.display()))?;
    match print_value(&fd, info) {
        Ok(()) => Ok(0),
        Err(err) => {
            eprintln!(
                "warning: set succeeded but readback for {} failed: {err:#}",
                info.name
            );
            Ok(1)
        }
    }
}

fn run(cli: Cli) -> anyhow::Result<i32> {
    match cli.command {
        Command::ListControls => {
            for control in CONTROLS {
                println!("{:<28} {}", control.name, control.description);
            }
            Ok(0)
        }
        Command::Get { control } => {
            let info = resolve_control(&control)?;
            handle_get(&cli.device, info)?;
            Ok(0)
        }
        Command::Set { control, value } => {
            let info = resolve_control(&control)?;
            handle_set(&cli.device, info, &value)
        }
        Command::GetFov => {
            let info = resolve_control("fov")?;
            handle_get(&cli.device, info)?;
            Ok(0)
        }
        Command::SetFov { value } => {
            let info = resolve_control("fov")?;
            handle_set(&cli.device, info, &value)
        }
    }
}

fn main() {
    let cli = Cli::parse();
    let code = match run(cli) {
        Ok(code) => code,
        Err(err) => {
            eprintln!("{err:#}");
            // Usage-class errors exit 2 like the reference tool; runtime errors exit 1.
            if matches!(
                err.downcast_ref::<Error>(),
                Some(Error::UnknownControl { .. })
            ) {
                2
            } else {
                1
            }
        }
    };
    exit(code);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_u8_accepts_full_range() {
        assert_eq!(parse_u8("0"), Some(0));
        assert_eq!(parse_u8("255"), Some(255));
    }

    #[test]
    fn parse_u8_rejects_out_of_range_and_garbage() {
        assert_eq!(parse_u8("256"), None);
        assert_eq!(parse_u8("-1"), None);
        assert_eq!(parse_u8("banana"), None);
        assert_eq!(parse_u8("1.5"), None);
    }

    #[test]
    fn parse_i32_accepts_full_range() {
        assert_eq!(parse_i32("-2147483648"), Some(i32::MIN));
        assert_eq!(parse_i32("2147483647"), Some(i32::MAX));
    }

    #[test]
    fn parse_i32_rejects_garbage() {
        assert_eq!(parse_i32("2147483648"), None);
        assert_eq!(parse_i32("x"), None);
        assert_eq!(parse_i32(""), None);
    }

    #[test]
    fn list_controls_formatting_is_aligned() {
        // Every row renders with a 28-char-padded name column, like C's %-28s.
        for control in CONTROLS {
            let row = format!("{:<28} {}", control.name, control.description);
            assert!(row.starts_with(control.name));
            assert!(row.len() >= 28 + 1 + control.description.len());
        }
    }
}
