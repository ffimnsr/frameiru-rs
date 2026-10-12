# Packaging support files

## One-click virtual camera (polkit)

`frameiru-ui` shows a status banner for the virtual camera (v4l2loopback) and
offers a **LOAD VIRTUAL CAMERA** button. The button runs the privileged helper
through `pkexec`; no terminal needed.

Two files make that work:

| File | Install to |
| --- | --- |
| `frameiru-loopback-helper` (built binary) | `/usr/libexec/frameiru-loopback-helper` (mode `0755`, owned by `root`) |
| `polkit/org.frameiru.loopback.policy` | `/usr/share/polkit-1/actions/org.frameiru.loopback.policy` (mode `0644`) |

```sh
sudo install -m755 target/release/frameiru-loopback-helper /usr/libexec/
sudo install -m644 packaging/polkit/org.frameiru.loopback.policy \
    /usr/share/polkit-1/actions/
```

After install, the button is one admin-password prompt away from
`modprobe v4l2loopback exclusive_caps=1 card_label=Frameiru video_nr=10`.
The helper is also idempotent: if the module is already loaded *with*
`exclusive_caps=1` it is a no-op; if loaded without it, it reloads the module
(and fails with a clear message when another app, e.g. OBS, holds the device).

### Security

- The helper is a std-only binary with a fixed argv whitelist
  (`frameiru-loopback-helper loopback-load` only) and never spawns a shell.
- The polkit action is restricted to that exact invocation via
  `org.freedesktop.policykit.exec.path` + `exec.argv1` annotations.
- `pkexec` runs the helper with a sanitized environment; the helper never
  reads user-controlled environment values.
- Without the installed policy file, `pkexec` falls back to the generic
  `org.freedesktop.policykit.exec` action (admin prompt) — same result,
  generic dialog.

### Development / testing

For uninstalled builds, the UI looks for the helper in this order:

1. `$FRAMEIRU_LOOPBACK_HELPER` (explicit override)
2. `/usr/libexec/frameiru-loopback-helper`
3. `/usr/local/libexec/frameiru-loopback-helper`
4. directory of the running `frameiru-ui` binary (e.g. `target/debug/`)

Build the helper alongside the UI:

```sh
cargo build -p frameiru-loopback-helper
cargo run -p frameiru-ui   # target/debug/frameiru-loopback-helper is found
```

or point directly at a scratch build:

```sh
FRAMEIRU_LOOPBACK_HELPER=/tmp/helper cargo run -p frameiru-ui
```

Without the polkit policy installed, the LOAD button still works — `pkexec`
shows the generic admin prompt instead of the branded one.

## Flatpak (GUI + CLI)

`packaging/flatpak/com.vastorigins.frameiru.yml` packages the Slint GUI and the
CLI as the Flatpak `com.vastorigins.frameiru`, together with what a sandboxed
app cannot download or elevate at runtime (ONNX Runtime) and the desktop
integration files that live next to it (`.desktop`, AppStream metainfo, icons).

```sh
# from the repository root
flatpak-builder --user --install --force-clean \
    --state-dir=.flatpak-builder build-dir \
    packaging/flatpak/com.vastorigins.frameiru.yml
flatpak run com.vastorigins.frameiru
```

The app module's source is `type: git` on `master`, so commit and push before
building; to build an uncommitted tree, point that source at the checkout
instead (`type: dir` + `path: ../..`).

### Cargo sources

Flatpak build sandboxes have no network, so crates are built `--offline`
against `packaging/flatpak/cargo-sources.json`, which vendors every crates.io
dependency of `Cargo.lock`. Regenerate it whenever `Cargo.lock` changes:

```sh
python3 -m venv .cargo-gen && .cargo-gen/bin/pip install aiohttp tomlkit pyyaml
curl -O https://raw.githubusercontent.com/flatpak/flatpak-builder-tools/master/cargo/flatpak-cargo-generator.py
.cargo-gen/bin/python flatpak-cargo-generator.py Cargo.lock \
    -o packaging/flatpak/cargo-sources.json
```

### ONNX Runtime

`ort` downloads onnxruntime from the network during a normal build, which the
sandbox forbids. Both binaries are therefore built with the `onnx-dynamic`
feature: onnxruntime is loaded at runtime via `dlopen`, from
`ORT_DYLIB_PATH=/app/lib/libonnxruntime.so` (set in the manifest's
`finish-args`). The `onnxruntime` module bundles Microsoft's v1.28.0 release
for `x86_64` and `aarch64` — the same upstream version `ort` 2.0.0-rc.13 would
download — so bumping `ort` means updating those two URLs and their `sha256`
values in the manifest.

### Virtual camera

`modprobe` and `pkexec` are host operations: a sandboxed app can neither load
`v4l2loopback` nor authenticate as an administrator, and the host's
`/usr/libexec/frameiru-loopback-helper` is not visible from inside the sandbox.
The module has to be loaded on the host, exactly as for a native install:

```sh
sudo modprobe v4l2loopback exclusive_caps=1 card_label="Frameiru" video_nr=10
```

or, with the host CLI installed, `sudo frameiru-cli setup` (persists across
reboots). The sandbox already grants `--device=all` (cameras and
`/dev/video10`) and `--device=dri` (GPU compositing), so once the module is
loaded the virtual camera works as usual. The banner's **LOAD VIRTUAL CAMERA**
button cannot elevate from inside the sandbox and reports the manual command
instead.

### Filesystem

The manifest grants no host filesystem access: the GUI picks background images
through the desktop portal, which hands the app read access to just that file.
Backgrounds referenced by path from the bundled CLI (e.g. `video:` files) need
an explicit grant, e.g.

```sh
flatpak override --user --filesystem=xdg-videos:ro com.vastorigins.frameiru
```

### Runtime

The manifest targets `org.freedesktop.Platform` **26.08** and builds with
`org.freedesktop.Sdk.Extension.rust-stable` (Rust 1.98). Both are easy to bump:
the `runtime-version` key only has to match an available
`org.freedesktop.Sdk`/[rust-stable](https://github.com/flathub/org.freedesktop.Sdk.Extension.rust-stable)
branch pair.

## Not included (yet)

- systemd unit for the pipeline daemon (planned). Note: the **boot auto-load** of the
  kernel module is already covered by `sudo frameiru-cli setup` (writes
  `/etc/modprobe.d/frameiru.conf` + `/etc/modules-load.d/frameiru.conf`); the polkit
  helper above only covers the current session.
- AppImage packaging. The Flatpak above covers the sandboxed install; note that
  `pkexec` + `modprobe` stay host operations either way.
- Flathub submission glue: a screenshot in the metainfo (Flathub requires at least
  one), a `flathub.json` in the Flathub repo for the supported arches, and a pinned
  `tag`/`commit` for the app source in the manifest.
