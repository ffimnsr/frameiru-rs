//! Consistency checks for the Flatpak packaging in `packaging/flatpak`.
//!
//! `flatpak-builder` is not available on every dev machine, so these tests only
//! validate the wiring that is easy to break when renaming or bumping things:
//! the app id used by every file, the paths the manifest installs, and the
//! vendored crates matching `Cargo.lock`.

use std::fs;
use std::path::{Path, PathBuf};

const APP_ID: &str = "com.vastorigins.frameiru";

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn read(rel: &str) -> String {
    let path = repo_root().join(rel);
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()))
}

/// One `[[package]]` block of `Cargo.lock`.
struct LockPackage {
    name: String,
    version: String,
    checksum: Option<String>,
}

/// `Cargo.lock` is plain TOML, but the tests run without a TOML dependency: the
/// fields are read from their own lines, which is how `cargo` writes them.
fn lock_packages(lock: &str) -> Vec<LockPackage> {
    let mut packages = Vec::new();
    for block in lock.split("[[package]]").skip(1) {
        let field = |key: &str| {
            block.lines().find_map(|l| {
                l.trim()
                    .strip_prefix(key)
                    .and_then(|rest| rest.strip_prefix(" = \""))
                    .and_then(|rest| rest.strip_suffix('"'))
                    .map(str::to_string)
            })
        };
        let (Some(name), Some(version)) = (field("name"), field("version")) else {
            continue;
        };
        packages.push(LockPackage {
            name,
            version,
            checksum: field("checksum"),
        });
    }
    packages
}

#[test]
fn app_id_is_consistent_across_packaging_files() {
    let manifest = read("packaging/flatpak/com.vastorigins.frameiru.yml");
    assert!(
        manifest.contains(&format!("id: {APP_ID}")),
        "manifest id must be {APP_ID}"
    );
    assert!(
        manifest.contains(&format!("{APP_ID}.desktop")),
        "manifest must install the {APP_ID} desktop file"
    );

    let desktop = read(&format!("packaging/flatpak/{APP_ID}.desktop"));
    assert!(desktop.contains(&format!("Icon={APP_ID}")));
    assert!(desktop.contains("Exec=frameiru-ui"));
    assert!(desktop.contains("Type=Application"));

    let metainfo = read(&format!("packaging/flatpak/{APP_ID}.metainfo.xml"));
    assert!(metainfo.contains(&format!("<id>{APP_ID}</id>")));
    assert!(metainfo.contains(&format!(
        "<launchable type=\"desktop-id\">{APP_ID}.desktop</launchable>"
    )));
    assert!(metainfo.contains("<project_license>"));

    for size in [128, 256] {
        let icon = repo_root().join(format!("packaging/flatpak/icons/{APP_ID}-{size}.png"));
        assert!(icon.is_file(), "missing icon {}", icon.display());
    }
}

#[test]
fn manifest_installs_existing_files_under_the_app_id() {
    let manifest = read("packaging/flatpak/com.vastorigins.frameiru.yml");
    let mut installed = 0;
    for line in manifest.lines().map(str::trim) {
        let Some(args) = line.strip_prefix("- install -Dm644 ") else {
            continue;
        };
        let mut parts = args.split_whitespace();
        let source = parts.next().expect("install needs a source path");
        let destination = parts.next().expect("install needs a destination path");
        assert!(
            repo_root().join(source).is_file(),
            "manifest installs missing file {source}"
        );
        let file_name = Path::new(destination)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default();
        assert!(
            file_name.starts_with(APP_ID),
            "installed desktop/metainfo/icon file must be named after the app id: {destination}"
        );
        installed += 1;
    }
    assert!(
        installed >= 4,
        "expected the manifest to install desktop, metainfo, and both icons"
    );
}

#[test]
fn vendored_cargo_sources_cover_the_lockfile() {
    let sources = read("packaging/flatpak/cargo-sources.json");
    let packages = lock_packages(&read("Cargo.lock"));
    let vendored = packages.iter().filter(|p| p.checksum.is_some()).count();
    assert!(vendored > 100, "unexpectedly few locked crates: {vendored}");

    for package in packages.iter().filter(|p| p.checksum.is_some()) {
        let name = &package.name;
        let version = &package.version;
        let checksum = package.checksum.as_deref().unwrap();
        // The generator writes one archive entry per crate, URL line followed
        // by its checksum: matching both at once keeps the check unambiguous.
        let entry = format!(
            "\"url\": \"https://static.crates.io/crates/{name}/{name}-{version}.crate\",\n        \"sha256\": \"{checksum}\","
        );
        assert!(
            sources.contains(&entry),
            "{name}-{version} is missing from packaging/flatpak/cargo-sources.json \
             (regenerate it after Cargo.lock changes, see packaging/README.md)"
        );
    }
}
