//! `setup`: one-time boot integration for the virtual camera kernel module.
//!
//! Writes `/etc/modprobe.d/frameiru.conf` and `/etc/modules-load.d/frameiru.conf`
//! so `v4l2loopback` loads at every boot with the Frameiru parameters
//! (`exclusive_caps=1 card_label="Frameiru" video_nr=10`). `--remove` undoes
//! it. Needs root; a failed write prints the `sudo` hint.
//!
//! The parameter strings intentionally mirror `frameiru-loopback-helper`'s
//! load command; keep them in sync when the module contract changes.

use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context as _, Result};

/// Kernel module name (also the `modules-load.d` unit).
pub(crate) const MODULE_NAME: &str = "v4l2loopback";
/// sysfs dir that exists while the module is loaded.
pub(crate) const SYSFS_MODULE: &str = "/sys/module/v4l2loopback";
/// sysfs bool-array parameter read when the module is loaded.
pub(crate) const SYSFS_EXCLUSIVE_CAPS: &str = "/sys/module/v4l2loopback/parameters/exclusive_caps";

const MODPROBE_PATH: &str = "/etc/modprobe.d/frameiru.conf";
const MODULES_LOAD_PATH: &str = "/etc/modules-load.d/frameiru.conf";

const MODPROBE_HEADER: &str = "# Frameiru virtual camera: load v4l2loopback with the\n\
# parameters the GUI and Zoom/Meet/Chrome expect.\n";
const MODPROBE_OPTIONS: &str =
    "options v4l2loopback exclusive_caps=1 card_label=\"Frameiru\" video_nr=10\n";
const MODULES_HEADER: &str = "# Load the v4l2loopback module at boot for Frameiru.\n";
const MODULES_UNIT: &str = "v4l2loopback\n";

/// Fixed boot-config paths; split out so tests can inject a scratch root.
#[derive(Debug, Clone)]
pub struct BootConfig {
    pub modprobe: PathBuf,
    pub modules: PathBuf,
}

impl Default for BootConfig {
    fn default() -> Self {
        Self {
            modprobe: PathBuf::from(MODPROBE_PATH),
            modules: PathBuf::from(MODULES_LOAD_PATH),
        }
    }
}

impl BootConfig {
    fn modprobe_dir(&self) -> &Path {
        self.modprobe.parent().expect("modprobe path has a parent")
    }
}

fn modprobe_content() -> Vec<u8> {
    format!("{MODPROBE_HEADER}{MODPROBE_OPTIONS}").into_bytes()
}

fn modules_content() -> Vec<u8> {
    format!("{MODULES_HEADER}{MODULES_UNIT}").into_bytes()
}

/// Writes `content` to `path`, creating parent dirs; reports whether the
/// file was touched. Fails with a sudo hint instead of a bare EACCES.
pub fn write_if_changed(path: &Path, content: &[u8]) -> Result<bool> {
    if path.exists() && fs::read(path).context("reading existing config")? == content {
        return Ok(false);
    }
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)
        .map_err(|e| {
            anyhow::anyhow!(
                "cannot write {}: {e}\nhint: run as root: `sudo frameiru-cli setup`",
                path.display()
            )
        })?;
    file.write_all(content)
        .with_context(|| format!("writing {}", path.display()))?;
    Ok(true)
}

/// Scans `/etc/modprobe.d` for other files that also configure
/// `v4l2loopback`; those can silently override (or clash with) our options.
pub fn scan_modprobe_conflicts(dir: &Path, own: &Path) -> Vec<String> {
    let mut warnings = Vec::new();
    let Ok(entries) = fs::read_dir(dir) else {
        return warnings;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path == own || path.extension().and_then(|e| e.to_str()) != Some("conf") {
            continue;
        }
        let Ok(content) = fs::read_to_string(&path) else {
            continue;
        };
        let line = content
            .lines()
            .map(str::trim)
            .find(|l| l.contains(MODULE_NAME) && !l.starts_with('#'));
        if let Some(line) = line {
            warnings.push(format!(
                "{} also configures {MODULE_NAME} (`{line}`); it may override \
                 the Frameiru options in {}",
                path.display(),
                own.display()
            ));
        }
    }
    warnings
}

/// True when every device in the module bool-array is exclusive (`Y`/`1`);
/// any unknown/missing token counts as non-exclusive (fail closed: the goal
/// is guaranteeing exclusive_caps, so garbage must never read as "fine").
/// `None` is never returned by this parser; it exists so callers can
/// distinguish "file unreadable" from "explicitly false".
pub(crate) fn parse_exclusive_caps(content: &str) -> Option<bool> {
    let mut seen = false;
    for token in content.split(',').map(str::trim).filter(|t| !t.is_empty()) {
        if !matches!(token, "Y" | "y" | "1" | "true" | "on") {
            return Some(false);
        }
        seen = true;
    }
    Some(seen)
}

fn module_exclusive_caps() -> Option<bool> {
    parse_exclusive_caps(&fs::read_to_string(SYSFS_EXCLUSIVE_CAPS).ok()?)
}

/// Installs the boot config (used by `frameiru-cli setup`). When the second
/// write fails, the first file is restored (or removed) so the boot state is
/// never left half-configured.
pub fn install(boot: &BootConfig) -> Result<()> {
    // Remember the previous modprobe.d content so a failed second write can
    // roll back exactly, preserving any hand edits.
    let prev_modprobe = fs::read(&boot.modprobe).ok();
    let modprobe_changed = write_if_changed(&boot.modprobe, &modprobe_content())?;
    let modules_changed = match write_if_changed(&boot.modules, &modules_content()) {
        Ok(changed) => changed,
        Err(e) => {
            let rollback = match &prev_modprobe {
                Some(prev) => fs::write(&boot.modprobe, prev)
                    .map(|_| format!("restored {}", boot.modprobe.display())),
                None => fs::remove_file(&boot.modprobe)
                    .map(|_| format!("removed {}", boot.modprobe.display())),
            };
            match rollback {
                Ok(note) => bail!("{e}\nrolled back: {note}"),
                Err(rollback_err) => bail!(
                    "{e}\nwarning: rollback of {} also failed: {rollback_err}",
                    boot.modprobe.display()
                ),
            }
        }
    };
    for warning in scan_modprobe_conflicts(boot.modprobe_dir(), &boot.modprobe) {
        eprintln!("warning: {warning}");
    }

    println!(
        "{}",
        if modprobe_changed {
            format!("wrote {}", boot.modprobe.display())
        } else {
            format!("ok: {} already configured", boot.modprobe.display())
        }
    );
    println!(
        "{}",
        if modules_changed {
            format!("wrote {}", boot.modules.display())
        } else {
            format!("ok: {} already configured", boot.modules.display())
        }
    );
    println!("ok: v4l2loopback will now load at every boot with exclusive_caps=1, card_label=\"Frameiru\", video_nr=10");

    match (Path::new(SYSFS_MODULE).is_dir(), module_exclusive_caps()) {
        (true, Some(true)) => println!("note: module is already loaded correctly for this session"),
        (true, Some(false)) => eprintln!(
            "warning: module is loaded WITHOUT exclusive_caps; it takes effect after a reload \
             (close apps using it, then: sudo rmmod v4l2loopback && sudo modprobe v4l2loopback \
             exclusive_caps=1 card_label=\"Frameiru\" video_nr=10) or use the frameiru-ui \
             LOAD VIRTUAL CAMERA button"
        ),
        (true, None) => {
            eprintln!("warning: module is loaded but its exclusive_caps flag is unreadable")
        }
        (false, _) => println!(
            "note: not loaded in this session yet — it loads at next boot, or now with:\n  \
             {} (the frameiru-ui LOAD VIRTUAL CAMERA button does the same)",
            session_modprobe_command()
        ),
    }
    Ok(())
}

/// Removes the boot config (used by `frameiru-cli setup --remove`).
pub fn remove(boot: &BootConfig) -> Result<()> {
    for path in [&boot.modprobe, &boot.modules] {
        match fs::remove_file(path) {
            Ok(()) => println!("removed {}", path.display()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                println!("ok: {} not present", path.display())
            }
            Err(e) => bail!("cannot remove {}: {e}", path.display()),
        }
    }
    println!("ok: v4l2loopback will no longer auto-load with Frameiru options at boot");
    Ok(())
}

/// The equivalent one-shot command, for the "load now" hint.
fn session_modprobe_command() -> String {
    format!("sudo modprobe {MODULE_NAME} exclusive_caps=1 card_label=\"Frameiru\" video_nr=10")
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU32, Ordering};

    use super::*;

    static NEXT_TMP: AtomicU32 = AtomicU32::new(0);

    /// Scratch dir unique per test: pid + monotonic counter (no timestamps).
    fn scratch_dir(tag: &str) -> PathBuf {
        let n = NEXT_TMP.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "frameiru-setup-test-{}-{}-{tag}",
            std::process::id(),
            n
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("create scratch dir");
        dir
    }

    fn boot_in(dir: &Path) -> BootConfig {
        BootConfig {
            modprobe: dir.join("modprobe.d").join("frameiru.conf"),
            modules: dir.join("modules-load.d").join("frameiru.conf"),
        }
    }

    #[test]
    fn install_writes_both_files_and_is_idempotent() {
        let dir = scratch_dir("install");
        let boot = boot_in(&dir);

        install(&boot).expect("install");
        assert_eq!(
            fs::read(&boot.modprobe).unwrap(),
            modprobe_content(),
            "modprobe.d content pinned"
        );
        assert_eq!(
            fs::read(&boot.modules).unwrap(),
            modules_content(),
            "modules-load.d content pinned"
        );

        // Second run: reported as unchanged, contents identical.
        assert!(!write_if_changed(&boot.modprobe, &modprobe_content()).unwrap());
        assert!(!write_if_changed(&boot.modules, &modules_content()).unwrap());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn remove_deletes_files_and_tolerates_missing() {
        let dir = scratch_dir("remove");
        let boot = boot_in(&dir);
        install(&boot).expect("install");
        remove(&boot).expect("remove");
        assert!(!boot.modprobe.exists() && !boot.modules.exists());
        // Idempotent.
        remove(&boot).expect("second remove");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn conflict_scan_flags_foreign_v4l2loopback_options_only() {
        let dir = scratch_dir("conflicts");
        let own = dir.join("frameiru.conf");
        fs::write(
            dir.join("obs.conf"),
            "options v4l2loopback exclusive_caps=0\n",
        )
        .unwrap();
        fs::write(dir.join("unrelated.conf"), "options some_module foo=1\n").unwrap();
        fs::write(
            dir.join("commented.conf"),
            "# options v4l2loopback exclusive_caps=0\n",
        )
        .unwrap();
        fs::write(dir.join("noext"), "options v4l2loopback x=1\n").unwrap();

        let warnings = scan_modprobe_conflicts(&dir, &own);
        assert_eq!(
            warnings.len(),
            1,
            "only the live foreign option warns: {warnings:?}"
        );
        assert!(warnings[0].contains("obs.conf"));

        // Our own file is never flagged.
        fs::write(&own, String::from_utf8(modprobe_content()).unwrap()).unwrap();
        let warnings = scan_modprobe_conflicts(&dir, &own);
        assert_eq!(warnings.len(), 1);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn exclusive_caps_requires_all_tokens_true() {
        assert_eq!(parse_exclusive_caps("Y"), Some(true));
        assert_eq!(parse_exclusive_caps("Y,Y"), Some(true));
        assert_eq!(parse_exclusive_caps("1,true\n"), Some(true));
        assert_eq!(parse_exclusive_caps("Y,N"), Some(false));
        assert_eq!(parse_exclusive_caps("N"), Some(false));
        assert_eq!(parse_exclusive_caps(""), Some(false));
        assert_eq!(parse_exclusive_caps("maybe"), Some(false));
        assert_eq!(parse_exclusive_caps("Y,maybe"), Some(false));
    }

    #[test]
    fn install_rolls_back_modprobe_when_modules_write_fails() {
        let dir = scratch_dir("rollback");
        let boot = BootConfig {
            modprobe: dir.join("modprobe.d").join("frameiru.conf"),
            // A regular file where the directory must be: create_dir_all fails.
            modules: dir.join("blocked").join("frameiru.conf"),
        };
        fs::write(dir.join("blocked"), "not a directory").unwrap();
        let modprobe_dir = boot.modprobe.parent().unwrap();
        fs::create_dir_all(modprobe_dir).unwrap();

        // Pre-existing hand-edited content must survive the rollback.
        let prev = b"# hand-edited options\n";
        fs::write(&boot.modprobe, prev).unwrap();
        let err = install(&boot).unwrap_err();
        assert!(
            err.to_string().contains("rolled back"),
            "error should mention the rollback: {err:?}"
        );
        assert_eq!(
            fs::read(&boot.modprobe).unwrap(),
            prev,
            "previous modprobe.d content restored"
        );

        // Fresh install (no prior file) must remove the half-written file.
        let boot2 = BootConfig {
            modprobe: dir.join("modprobe.d").join("fresh.conf"),
            modules: dir.join("blocked").join("frameiru.conf"),
        };
        let err = install(&boot2).unwrap_err();
        assert!(err.to_string().contains("removed"));
        assert!(!boot2.modprobe.exists(), "fresh modprobe file rolled back");
        let _ = fs::remove_dir_all(&dir);
    }
}
