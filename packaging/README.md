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

## Not included (yet)

- systemd unit for the pipeline daemon (planned). Note: the **boot auto-load** of the
  kernel module is already covered by `sudo frameiru-cli setup` (writes
  `/etc/modprobe.d/frameiru.conf` + `/etc/modules-load.d/frameiru.conf`); the polkit
  helper above only covers the current session.
- Flatpak/AppImage sandboxing notes: `pkexec` + `modprobe` are host
  operations; a sandboxed UI needs the host-side helper and policy installed
  by the package.