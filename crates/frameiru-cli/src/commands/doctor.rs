//! `doctor`: preflight checks for the virtual camera stack, with exact fixes.
//!
//! Feature-independent: everything is read from sysfs, `/proc`, `/etc`, and
//! PATH — no `v4l2` feature needed, so it works on minimal CLI builds.
//! `--fix` retries the privileged module step through the polkit helper
//! (same binary the GUI uses) when one is installed.
//!
//! Exit code: 0 when nothing failed, 1 when any check failed.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::Result;

use super::setup::{self, MODULE_NAME, SYSFS_MODULE};

const EBUSY: i32 = 16;
const EACCES: i32 = 13;

/// One doctor result line.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Check {
    label: &'static str,
    level: Level,
    detail: String,
    fix: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Level {
    Ok,
    Warn,
    Skip,
    Fail,
}

impl Level {
    fn is_fail(self) -> bool {
        matches!(self, Level::Fail)
    }
}

/// Privileged fix attempts (module load/reload) happen here; the helper is
/// the same fixed-argv binary the GUI uses via `pkexec`.
fn find_loopback_helper() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("FRAMEIRU_LOOPBACK_HELPER") {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Some(path);
        }
    }
    [
        PathBuf::from("/usr/libexec/frameiru-loopback-helper"),
        PathBuf::from("/usr/local/libexec/frameiru-loopback-helper"),
    ]
    .into_iter()
    .find(|p| p.is_file())
}

/// Runs every check; `fix` retries the privileged module step afterwards.
/// Returns the process exit code (0 = all checks passed).
pub fn run(fix: bool) -> Result<i32> {
    if fix {
        let fixable = run_checks(&setup::BootConfig::default())
            .iter()
            .any(|c| c.level.is_fail() && c.fix == "module");
        if fixable {
            match &find_loopback_helper() {
                Some(helper) => {
                    println!(
                        "--fix: running {} loopback-load via pkexec …",
                        helper.display()
                    );
                    let output = Command::new("pkexec")
                        .arg(helper)
                        .arg("loopback-load")
                        .output()
                        .map_err(|e| anyhow::anyhow!("cannot start pkexec: {e}"))?;
                    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
                    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
                    let detail = if stderr.is_empty() { stdout } else { stderr };
                    if output.status.success() {
                        println!("--fix: ok — {detail}");
                    } else {
                        let code = output.status.code().unwrap_or(-1);
                        eprintln!(
                            "--fix: failed (exit {code}): {detail}\n\
                             hint: close other apps using v4l2loopback (e.g. OBS), \
                             or run: sudo modprobe {MODULE_NAME} exclusive_caps=1 \
                             card_label=\"Frameiru\" video_nr=10"
                        );
                    }
                }
                None => eprintln!(
                    "--fix: privileged helper not installed (see packaging/README.md); \
                     run manually:\n  sudo modprobe {MODULE_NAME} exclusive_caps=1 \
                     card_label=\"Frameiru\" video_nr=10"
                ),
            }
        }
    }

    // Re-run after --fix so the report reflects the new state.
    let checks = run_checks(&setup::BootConfig::default());
    let failed = print_report(&checks);
    Ok(if failed { 1 } else { 0 })
}

/// Prints the report and returns whether any check failed.
fn print_report(checks: &[Check]) -> bool {
    let mut failed = false;
    for check in checks {
        let mark = match check.level {
            Level::Ok => "[ok]  ",
            Level::Warn => "[warn]",
            Level::Skip => "[skip]",
            Level::Fail => "[fail]",
        };
        println!("{mark} {}", check.label);
        if !check.detail.is_empty() {
            println!("        {}", check.detail);
        }
        let fix = if check.fix == "module" {
            format!(
                "load the module: sudo modprobe {MODULE_NAME} exclusive_caps=1 \
                 card_label=\"Frameiru\" video_nr=10 (or `frameiru-cli doctor --fix`, \
                 or the GUI LOAD VIRTUAL CAMERA button)"
            )
        } else {
            check.fix.clone()
        };
        if !fix.is_empty() {
            println!("        fix: {fix}");
        }
        failed |= check.level.is_fail();
    }
    if failed {
        println!("\n1 or more checks failed — see the fix lines above.");
    } else {
        println!("\nall checks passed.");
    }
    failed
}

fn run_checks(boot: &setup::BootConfig) -> Vec<Check> {
    let mut checks = Vec::new();

    // 1. Kernel module loaded.
    let module_loaded = Path::new(SYSFS_MODULE).is_dir();
    checks.push(Check {
        label: "v4l2loopback kernel module loaded",
        level: if module_loaded {
            Level::Ok
        } else {
            Level::Fail
        },
        detail: if module_loaded {
            String::new()
        } else {
            "not loaded in this session".into()
        },
        fix: if module_loaded {
            String::new()
        } else {
            "module".into()
        },
    });

    // Enumerated once: checks 2 and 3 both need the loopback node set.
    let nodes = sysfs_loopback_nodes(Path::new("/sys/class/video4linux"));

    // 2. exclusive_caps of the virtual camera node (only meaningful when the
    // module is loaded).
    if module_loaded {
        // Prefer the Frameiru-labelled node; the official load always creates
        // the camera on `video_nr=10`, so that number is the honest fallback.
        let target = nodes
            .iter()
            .find(|(_, name)| name == setup::CARD_LABEL)
            .map_or(setup::VIDEO_NR, |(nr, _)| *nr);
        let exclusive = setup::module_exclusive_caps_for(target);
        checks.push(Check {
            label: "exclusive_caps=1 (visible to Chrome/Zoom)",
            level: if exclusive == Some(true) {
                Level::Ok
            } else {
                Level::Fail
            },
            detail: match exclusive {
                Some(true) => String::new(),
                Some(false) => format!(
                    "/dev/video{target}: device is not in exclusive_caps mode; reload required"
                ),
                None => format!(
                    "/dev/video{target}: exclusive_caps could not be confirmed; reload with \
                     the Frameiru options"
                ),
            },
            fix: if exclusive == Some(true) {
                String::new()
            } else {
                "module".into()
            },
        });
    } else {
        checks.push(Check {
            label: "exclusive_caps",
            level: Level::Skip,
            detail: "module not loaded".into(),
            fix: String::new(),
        });
    }

    // 3. Loopback node present in sysfs (driver link), plus its card name.
    if nodes.is_empty() {
        let level = if module_loaded {
            Level::Fail
        } else {
            Level::Skip
        };
        checks.push(Check {
            label: "loopback video node (/dev/videoN)",
            level,
            detail: if module_loaded {
                "module loaded but no v4l2loopback device visible".into()
            } else {
                "module not loaded".into()
            },
            fix: if module_loaded {
                "module".into()
            } else {
                String::new()
            },
        });
    } else {
        let (nr, name) = &nodes[0];
        let dev = format!("/dev/video{nr}");
        let detail = format!("{dev} ({name})");
        checks.push(Check {
            label: "loopback video node",
            level: Level::Ok,
            detail,
            fix: String::new(),
        });

        // 3b. The node set must be openable by this user: any node opening
        // read-write passes (a busy first node is not a fault when a later
        // one is fine); first error classifies the failure otherwise.
        let mut any_open = false;
        let mut first_error: Option<std::io::Error> = None;
        for (nr, _name) in &nodes {
            match std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(format!("/dev/video{nr}"))
            {
                Ok(_) => {
                    any_open = true;
                    break;
                }
                Err(e) => {
                    first_error.get_or_insert(e);
                }
            }
        }
        if any_open {
            checks.push(Check {
                label: "loopback node openable",
                level: Level::Ok,
                detail: "at least one loopback node opened read-write".into(),
                fix: String::new(),
            });
        } else {
            let e = first_error.expect("at least one node was tried");
            match e.raw_os_error() {
                Some(EACCES) => checks.push(Check {
                    label: "loopback node openable",
                    level: Level::Fail,
                    detail: "/dev/video*: permission denied".to_string(),
                    fix: "add your user to the `video` group and re-login:\n            \
                          sudo usermod -aG video $USER"
                        .to_string(),
                }),
                // EBUSY only means another app holds it; not a setup fault.
                Some(EBUSY) => checks.push(Check {
                    label: "loopback node openable",
                    level: Level::Ok,
                    detail: "/dev/video* busy (another app holds every node)".into(),
                    fix: String::new(),
                }),
                _ => checks.push(Check {
                    label: "loopback node openable",
                    level: Level::Warn,
                    detail: format!("/dev/video*: {e}"),
                    fix: String::new(),
                }),
            }
        }
    }

    // 4. video group membership (device permissions for the whole stack).
    checks.push(check_video_group());

    // 5. ffmpeg/ffprobe (optional: only for `video:<path>` backgrounds).
    let missing_tools: Vec<String> = ["ffmpeg", "ffprobe"]
        .iter()
        .filter(|tool| find_on_path(tool, std::env::var("PATH").ok().as_deref()).is_none())
        .map(|t| t.to_string())
        .collect();
    checks.push(Check {
        label: "ffmpeg + ffprobe (video backgrounds)",
        level: if missing_tools.is_empty() {
            Level::Ok
        } else {
            Level::Warn
        },
        detail: if missing_tools.is_empty() {
            "both on PATH".into()
        } else {
            format!("missing: {}", missing_tools.join(", "))
        },
        fix: if missing_tools.is_empty() {
            String::new()
        } else {
            "install ffmpeg (only needed when using a video:<path> background)".into()
        },
    });

    // 6. Boot auto-load config (frameiru-cli setup).
    let boot_installed = boot.modprobe.is_file() && boot.modules.is_file();
    checks.push(Check {
        label: "boot auto-load configured",
        level: if boot_installed {
            Level::Ok
        } else {
            Level::Warn
        },
        detail: if boot_installed {
            format!(
                "{} + {} present",
                boot.modprobe.display(),
                boot.modules.display()
            )
        } else {
            "module won't load at boot on its own".into()
        },
        fix: if boot_installed {
            String::new()
        } else {
            "sudo frameiru-cli setup (one-time; --remove undoes it)".into()
        },
    });

    checks
}

/// Enumerates v4l2loopback nodes from the sysfs class dir; each entry needs
/// `videoN/device/driver` to point into the v4l2loopback driver.
fn sysfs_loopback_nodes(class_dir: &Path) -> Vec<(u32, String)> {
    let mut nodes = Vec::new();
    let Ok(entries) = std::fs::read_dir(class_dir) else {
        return nodes;
    };
    for entry in entries.flatten() {
        let dir = entry.path();
        let Some(name) = dir.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let Some(nr) = name
            .strip_prefix("video")
            .and_then(|rest| rest.parse::<u32>().ok())
        else {
            continue;
        };
        let driver = dir.join("device").join("driver");
        let is_loopback = std::fs::read_link(&driver)
            .map(|target| target.to_string_lossy().contains("v4l2loopback"))
            .unwrap_or(false);
        if !is_loopback {
            continue;
        }
        let card = std::fs::read_to_string(dir.join("name"))
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|_| "(unreadable)".into());
        nodes.push((nr, card));
    }
    nodes.sort_by_key(|(nr, _)| *nr);
    nodes
}

/// Membership in the `video` group: the user's gid list (from
/// `/proc/self/status`) must contain the group's gid, or the group's member
/// list must contain the user's name. Reads are relative to `proc_status` /
/// `etc_group` so tests can inject fixtures.
fn check_video_group() -> Check {
    let proc_status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let etc_group = std::fs::read_to_string("/etc/group").unwrap_or_default();
    let username = std::env::var("USER")
        .ok()
        .filter(|u| !u.is_empty())
        .unwrap_or_else(|| "".into());

    let groups_line = proc_status
        .lines()
        .find(|l| l.starts_with("Groups:"))
        .unwrap_or("");
    let my_gids: Vec<u64> = groups_line
        .split_whitespace()
        .skip(1)
        .filter_map(|g| g.parse().ok())
        .collect();

    let video_line = etc_group
        .lines()
        .find(|l| l.starts_with("video:"))
        .unwrap_or("");
    let (video_gid, members) = parse_group_line(video_line);

    let member = match video_gid {
        Some(gid) => {
            // Either the process's gid set contains the group, or the group's
            // member list names us (the gid set can lag behind usermod until
            // re-login, and the member list is authoritative for the next
            // session).
            my_gids.contains(&gid) || (!username.is_empty() && members.contains(&username))
        }
        None => {
            // No video group on this system at all: nothing to join.
            return Check {
                label: "user in `video` group",
                level: Level::Skip,
                detail: "no `video` group on this system".into(),
                fix: String::new(),
            };
        }
    };

    Check {
        label: "user in `video` group",
        level: if member { Level::Ok } else { Level::Fail },
        detail: if member {
            "group membership confirmed".into()
        } else {
            format!("user {username:?} not in `video` ({video_gid:?})")
        },
        fix: if member {
            String::new()
        } else {
            "sudo usermod -aG video $USER, then re-login".into()
        },
    }
}

/// `video:x:44:alice,bob` -> (gid, member list).
fn parse_group_line(line: &str) -> (Option<u64>, Vec<String>) {
    let mut parts = line.trim().split(':');
    let _name = parts.next();
    let _passwd = parts.next();
    let gid = parts.next().and_then(|g| g.parse().ok());
    let members = parts
        .next()
        .unwrap_or("")
        .split(',')
        .filter(|m| !m.is_empty())
        .map(str::to_string)
        .collect();
    (gid, members)
}

/// Finds `tool` in the PATH directories (or the explicit `path`).
fn find_on_path(tool: &str, path: Option<&str>) -> Option<PathBuf> {
    let path = path.unwrap_or("/usr/local/bin:/usr/bin:/bin");
    path.split(':')
        .filter(|d| !d.is_empty())
        .map(|d| Path::new(d).join(tool))
        .find(|p| p.is_file())
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU32, Ordering};

    use super::*;

    static NEXT_TMP: AtomicU32 = AtomicU32::new(0);

    fn test_dir(tag: &str) -> PathBuf {
        let n = NEXT_TMP.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "frameiru-doctor-test-{}-{}-{tag}",
            std::process::id(),
            n
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        dir
    }

    #[test]
    fn parses_video_group_line() {
        assert_eq!(
            parse_group_line("video:x:44:alice,bob"),
            (Some(44), vec!["alice".into(), "bob".into()])
        );
        assert_eq!(
            parse_group_line("video:x:44:"),
            (Some(44), Vec::<String>::new())
        );
        assert_eq!(parse_group_line("nonsense"), (None, vec![]));
    }

    #[test]
    fn finds_tools_only_in_existing_dirs() {
        let exe = std::env::current_exe().expect("current exe");
        let dir = exe.parent().expect("exe dir");
        let name = exe.file_name().and_then(|n| n.to_str()).expect("exe name");
        let found = find_on_path(name, Some(dir.to_str().unwrap()));
        assert_eq!(found, Some(exe.clone()));
        assert_eq!(
            find_on_path("definitely-not-a-tool", Some("/nonexistent")),
            None
        );
        assert_eq!(
            find_on_path(name, None),
            None,
            "default PATH never has our exe"
        );
    }

    #[test]
    fn sysfs_discovery_reads_driver_links_and_names() {
        let base = test_dir("sysfs");
        let video = base.join("video10");
        std::fs::create_dir_all(video.join("device")).unwrap();
        std::fs::write(video.join("name"), "Frameiru\n").unwrap();
        std::os::unix::fs::symlink(
            "../../bus/platform/drivers/v4l2loopback",
            video.join("device").join("driver"),
        )
        .unwrap();
        std::fs::create_dir_all(base.join("video9").join("device")).unwrap();
        std::fs::write(base.join("video9/name"), "Webcam\n").unwrap();
        // video9 has no driver link -> not a loopback node.
        let unrelated = base.join("video2");
        std::fs::create_dir_all(unrelated.join("device")).unwrap();
        std::fs::write(unrelated.join("name"), "Not Loopback\n").unwrap();
        std::os::unix::fs::symlink(
            "../../usb/drivers/uvcvideo",
            unrelated.join("device").join("driver"),
        )
        .unwrap();

        assert_eq!(
            sysfs_loopback_nodes(&base),
            vec![(10, "Frameiru".to_string())]
        );
        let _ = std::fs::remove_dir_all(&base);
    }
}
