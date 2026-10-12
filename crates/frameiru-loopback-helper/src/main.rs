//! Privileged helper: load `v4l2loopback` for the Frameiru virtual camera.
//!
//! Runs as root, normally through `pkexec`, so the GUI can enable the virtual
//! camera without a terminal. It can do exactly one thing (`loopback-load`)
//! and validates that strictly: the argument list is fixed, kernel module
//! tools are executed directly (never through a shell), and every parameter
//! is a constant.
//!
//! Exit codes: 0 success, 1 operation failed, 2 usage error.

use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

pub const MODULE_NAME: &str = "v4l2loopback";
pub const SYSFS_MODULE: &str = "/sys/module/v4l2loopback";
pub const SYSFS_EXCLUSIVE_CAPS: &str = "/sys/module/v4l2loopback/parameters/exclusive_caps";
pub const CARD_LABEL: &str = "Frameiru";
pub const VIDEO_NR: &str = "10";

/// Where `modprobe`/`rmmod` live on typical distros; the bare name is a
/// PATH fallback (pkexec ships a minimal but sane PATH).
const TOOL_DIRS: &[&str] = &["/sbin", "/usr/sbin", "/bin", "/usr/bin"];

/// The one request this helper accepts.
#[derive(Debug, PartialEq, Eq)]
enum Request {
    LoopbackLoad,
}

fn parse_args(args: &[String]) -> Result<Request, String> {
    match args {
        [op] if op == "loopback-load" => Ok(Request::LoopbackLoad),
        _ => Err(format!(
            "unsupported request {args:?} (this helper may only run: loopback-load)"
        )),
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let request = match parse_args(&args) {
        Ok(request) => request,
        Err(e) => {
            eprintln!("{e}");
            eprintln!("usage: frameiru-loopback-helper loopback-load");
            return ExitCode::from(2);
        }
    };
    match request {
        Request::LoopbackLoad => match ensure_loaded() {
            Ok(()) => {
                println!(
                    "OK: {MODULE_NAME} loaded with exclusive_caps=1, \
                     card_label={CARD_LABEL}, video_nr={VIDEO_NR}"
                );
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("ERROR: {e}");
                ExitCode::from(1)
            }
        },
    }
}

/// Loads the module with the Frameiru parameters, idempotently: already
/// loaded with `exclusive_caps` -> no-op; loaded without it -> reload.
fn ensure_loaded() -> Result<(), String> {
    let mut reload_note = String::new();
    if Path::new(SYSFS_MODULE).is_dir() {
        if exclusive_caps_enabled() {
            return Ok(());
        }
        // Loaded with exclusive_caps=0 (/etc/modprobe.d, manual modprobe):
        // try to unload so the reload takes effect. rmmod may fail because
        // another app holds the devices (OBS, a second Frameiru) — that is
        // not fatal yet, the modprobe + verification below decides.
        if let Err(e) = run_tool("rmmod", &[MODULE_NAME]) {
            reload_note = format!(" (reload blocked: {e})");
        }
    }
    let label = format!("card_label={CARD_LABEL}");
    let nr = format!("video_nr={VIDEO_NR}");
    run_tool("modprobe", &[MODULE_NAME, "exclusive_caps=1", &label, &nr])?;
    wait_for_video_node();
    if !exclusive_caps_enabled() {
        return Err(format!(
            "{MODULE_NAME} loaded but exclusive_caps did not take effect{reload_note}; \
             close other applications using it (e.g. OBS) and check /etc/modprobe.d \
             for conflicting v4l2loopback options"
        ));
    }
    Ok(())
}

/// True when every device in the module-wide bool-array parameter is
/// exclusive (`Y`/`1`). Unreadable or empty files read as "not enabled"
/// here because this helper's only job is to guarantee the right config.
fn exclusive_caps_enabled() -> bool {
    std::fs::read_to_string(SYSFS_EXCLUSIVE_CAPS)
        .map(|content| exclusive_caps_content(&content))
        .unwrap_or(false)
}

fn exclusive_caps_content(content: &str) -> bool {
    let mut seen = false;
    for token in content.split(',').map(str::trim).filter(|t| !t.is_empty()) {
        if !matches!(token, "Y" | "y" | "1" | "true" | "on") {
            return false;
        }
        seen = true;
    }
    seen
}

fn video_node_path() -> PathBuf {
    Path::new("/dev").join(format!("video{VIDEO_NR}"))
}

/// udev may create the node asynchronously after `modprobe` returns; wait a
/// short bounded time for it.
fn wait_for_video_node() {
    let path = video_node_path();
    for _ in 0..40 {
        if path.exists() {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
}

fn tool_candidates(name: &str) -> Vec<PathBuf> {
    let mut paths: Vec<PathBuf> = TOOL_DIRS.iter().map(|d| Path::new(d).join(name)).collect();
    paths.push(PathBuf::from(name));
    paths
}

fn first_existing(candidates: &[PathBuf]) -> Option<PathBuf> {
    candidates.iter().find(|p| p.is_file()).cloned()
}

fn run_tool(name: &str, args: &[&str]) -> Result<(), String> {
    let tool = first_existing(&tool_candidates(name))
        .ok_or_else(|| format!("cannot find {name} in {} or PATH", TOOL_DIRS.join(", ")))?;
    // Explicit argv, no shell: this is the trusted invocation surface of a
    // root process, so nothing from the environment may flow into it.
    let output = Command::new(&tool)
        .args(args)
        .output()
        .map_err(|e| format!("cannot execute {}: {e}", tool.display()))?;
    if output.status.success() {
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let detail = match (stdout.as_str(), stderr.as_str()) {
        (s, "") => s.to_string(),
        ("", s) => s.to_string(),
        (s, e) => format!("{s}: {e}"),
    };
    Err(if detail.is_empty() {
        format!("{} exited with {}", tool.display(), output.status)
    } else {
        format!("{}: {detail}", tool.display())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_exactly_loopback_load() {
        assert_eq!(
            parse_args(&["loopback-load".into()]),
            Ok(Request::LoopbackLoad)
        );
        assert!(parse_args(&[]).is_err());
        assert!(parse_args(&["loopback-load".into(), "extra".into()]).is_err());
        assert!(parse_args(&["rmmod".into()]).is_err());
        assert!(parse_args(&["-h".into()]).is_err());
    }

    #[test]
    fn exclusive_caps_requires_all_devices_exclusive() {
        assert!(exclusive_caps_content("Y"));
        assert!(exclusive_caps_content("Y,Y"));
        assert!(exclusive_caps_content("1,true\n"));
        assert!(!exclusive_caps_content("N"));
        assert!(!exclusive_caps_content("Y,N"));
        assert!(!exclusive_caps_content(""));
        assert!(!exclusive_caps_content("maybe"));
    }

    #[test]
    fn tool_candidates_cover_standard_dirs_then_path() {
        let paths = tool_candidates("modprobe");
        assert_eq!(paths.len(), TOOL_DIRS.len() + 1);
        assert_eq!(paths[0], Path::new("/sbin").join("modprobe"));
        assert_eq!(*paths.last().unwrap(), PathBuf::from("modprobe"));
    }

    #[test]
    fn first_existing_finds_the_current_executable_only() {
        let exe = std::env::current_exe().expect("current exe");
        assert_eq!(
            first_existing(&[PathBuf::from("/nonexistent-tool"), exe.clone()]),
            Some(exe)
        );
        assert_eq!(first_existing(&[PathBuf::from("/nonexistent-tool")]), None);
    }
}
