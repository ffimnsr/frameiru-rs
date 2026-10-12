# Releasing Frameiru

Frameiru ships prebuilt release tarballs on GitHub. This page documents the
release process end to end: how to cut a release, what the CI/CD workflows
check and produce, and how to install and verify the artifacts.

## Release process

1. **Bump the version** in the workspace manifest (`version` under
   `[workspace.package]` in `Cargo.toml`). All workspace crates inherit it.
2. **Commit and push to `master`.** CI (`ci.yml`) runs on every push and pull
   request; wait for it to go green — the release pipeline trusts that signal.
3. **Tag the commit** with a `v`-prefixed tag matching the manifest version:

   ```sh
   git tag v0.1.0
   git push origin v0.1.0
   ```

4. The `Release` workflow (`release.yml`) takes over:

   - **verify** — reruns quality gates (fmt, clippy, rustdoc). The tagged
     commit's CI run is looked up via the GitHub API; if it is already fully
     green the test suite is skipped (no duplicate run), otherwise tests run
     here. It also signs and attests `install.sh` and `Cargo.lock`.
   - **build** — builds release binaries for `x86_64-unknown-linux-gnu`
     (native `ubuntu-latest`) and `aarch64-unknown-linux-gnu` (native
     `ubuntu-24.04-arm`), packages one tarball per target, and signs,
     attests, and SBOMs each archive.
   - **publish** — only on real tags (`workflow_dispatch` runs stop after
     build, so you can dry-run the pipeline without publishing). Attaches all
     artifacts to the GitHub release and generates release notes from the
     changelog of commits since the previous tag.

> Version mismatch between tag and manifest is not enforced automatically; the
> tag name wins for artifact naming (`frameiru-<tag>-<target>.tar.gz`). Keep
> them in sync manually.

## Artifact inventory

Each release contains:

| Artifact | Description |
| --- | --- |
| `frameiru-<version>-<target>.tar.gz` | User install payload (see below) |
| `*.tar.gz.sha256` | SHA-256 checksum of the tarball |
| `*.tar.gz.sigstore.json` | cosign signature bundle of the tarball |
| `*.spdx.json` (+ `.sigstore.json`) | SBOM of the package contents |
| `install.sh` (+ `.sha256`, `.sigstore.json`) | Official installer |
| `Cargo.lock` (+ `.sha256`, `.sigstore.json`) | Pinned dependency snapshot |
| SLSA provenance | `actions/attest-build-provenance`, listed under the release's attestation tab |

Tarball layout (what the installer extracts):

```
frameiru-<version>-<target>/
├── frameiru-cli                # v4l2-loopback pipeline CLI (full features)
├── frameiru-ui                 # Slint desktop GUI
├── frameiru-webcam-utils       # Anker PowerConf C200 UVC utilities
├── frameiru-loopback-helper    # pkexec-friendly v4l2loopback loader
├── packaging/polkit/
│   └── org.frameiru.loopback.policy
├── README.md
└── LICENSE
```

## Installing

```sh
curl -fsSL https://raw.githubusercontent.com/ffimnsr/frameiru-rs/master/install.sh | sh
```

See `install.sh --help` for options: `--version` to pin a tag,
`--bin-dir`/`--doc-dir`/`--license-dir` to relocate, and `--with-polkit` to
additionally install the loopback helper and polkit policy to the fixed
system paths (`/usr/libexec`, `/usr/share/polkit-1/actions`) so the GUI's
LOAD VIRTUAL CAMERA button shows the branded admin prompt.

## Verifying a release

Checksums are published next to every artifact; the installer always verifies
the tarball's SHA-256 before extracting.

```sh
# Manual checksum check
curl -fsSL -O https://github.com/ffimnsr/frameiru-rs/releases/latest/download/frameiru-0.1.0-x86_64-unknown-linux-gnu.tar.gz
curl -fsSL -O https://github.com/ffimnsr/frameiru-rs/releases/latest/download/frameiru-0.1.0-x86_64-unknown-linux-gnu.tar.gz.sha256
sha256sum -c frameiru-0.1.0-x86_64-unknown-linux-gnu.tar.gz.sha256
```

Signature bundles are cosign `sign-blob` bundles verifiable against the
GitHub OIDC issuer; SLSA provenance is verifiable with
[`gh attestation verify`](https://cli.github.com/manual/gh_attestation_verify):

```sh
gh attestation verify frameiru-0.1.0-x86_64-unknown-linux-gnu.tar.gz \
  --repo ffimnsr/frameiru-rs
```

## Local dry run

The packaging steps are reproducible locally so you can test the whole flow
before tagging:

```sh
mk release-package   # builds the four binaries and packs dist/*.tar.gz
sh install.sh --bin-dir "$PWD/dist/install-test" \
  --version 0.1.0    # or: FRAMEIRU_INSTALL_LOCAL_PACKAGE=dist/*.tar.gz sh install.sh
```

Use the `FRAMEIRU_INSTALL_LOCAL_PACKAGE` environment variable to point the
installer at a local tarball — no network needed.

## Notes

- **System dependencies on builders**: `frameiru-ui` links against fontconfig
  at build time (slint), so CI installs `libfontconfig1-dev`. At runtime the
  GUI additionally needs `libfontconfig.so.1`; the installer warns when it is
  missing.
- **Feature sets**: the released CLI is built with `--features full`
  (gpu + onnx + v4l2 + dbus); the GUI uses its default features. ONNX Runtime
  is linked statically via `ort`'s `download-binaries`, so the binaries carry
  no extra `.so` payload.
- **Reproducibility**: builds pin `SOURCE_DATE_EPOCH` to the tagged commit
  time, use thin LTO/1 codegen unit/strip/panic-abort release profiles, and
  remap build paths (`RUSTFLAGS=--remap-path-prefix=...`).
- **Publishing embargo**: `publish` only runs for `github.ref_type == 'tag'`;
  `workflow_dispatch` runs verify + build so a release can be smoke-tested
  without publishing anything.