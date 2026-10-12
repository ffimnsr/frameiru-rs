#!/bin/sh
# shellcheck shell=dash
# shellcheck disable=SC3043

# Official Frameiru installer.
# Runs on Unix-like shells and installs the latest Frameiru release from
# GitHub. Linux-only: Frameiru publishes to v4l2loopback and targets
# x86_64 / aarch64 gnu targets.

REPO_OWNER="ffimnsr"
REPO_NAME="frameiru-rs"
PACKAGE_NAME="frameiru"
BINARIES="frameiru-cli frameiru-ui frameiru-webcam-utils frameiru-loopback-helper"
LOOPBACK_HELPER="frameiru-loopback-helper"
POLKIT_POLICY="packaging/polkit/org.frameiru.loopback.policy"

main() {
    if [ "${KSH_VERSION-}" = 'Version JM 93t+ 2010-03-05' ]; then
        err 'installer does not work with this ksh93 version; please try bash'
    fi

    set -u
    parse_args "$@"

    local _arch
    if [ -n "${ARCH:-}" ]; then
        _arch="${ARCH}"
    else
        # get_architecture errs with a specific message on unsupported
        # hardware; the subshell exit status propagates through $() so a
        # bare `|| exit 1` is enough (no ensure/assert_nz double-report).
        _arch="$(get_architecture)" || exit 1
    fi
    echo "Detected architecture: ${_arch}"

    local _tmp_dir
    _tmp_dir="$(mktemp -d)" || err "mktemp: could not create temporary directory"
    cd "${_tmp_dir}" || err "cd: failed to enter directory: ${_tmp_dir}"
    echo "Temporary directory: ${_tmp_dir}"

    local _package
    if ! _package="$(download_package "${_arch}")" || [ -z "${_package}" ]; then
        err "failed to obtain a ${PACKAGE_NAME} release package for ${_arch}"
    fi
    echo "Downloaded package: ${_package}"

    maybe_verify_checksum "${_package}"

    case "${_package}" in
        *.tar.gz) need_cmd tar; ensure tar -xf "${_package}" ;;
        *) err "unsupported package format: ${_package}" ;;
    esac

    local _package_root
    _package_root=$(basename "${_package}" | sed -E 's/\.(tar\.gz)$//')

    install_binaries "${_package_root}"

    if [ "${INSTALL_POLKIT}" = "1" ]; then
        install_system_loopback "${_package_root}"
    else
        print_polkit_hint
    fi

    install_docs "${_package_root}"
    install_license "${_package_root}"

    check_runtime_deps
    final_message
}

parse_args() {
    BIN_DIR_DEFAULT="${HOME}/.local/bin"
    DATA_HOME_DEFAULT="${XDG_DATA_HOME:-${HOME}/.local/share}"
    DOC_DIR_DEFAULT="${DATA_HOME_DEFAULT}/doc"
    LIC_DIR_DEFAULT="${DATA_HOME_DEFAULT}/licenses"
    SUDO_DEFAULT="sudo"
    BIN_DIR="${BIN_DIR_DEFAULT}"
    DOC_DIR="${DOC_DIR_DEFAULT}"
    LIC_DIR="${LIC_DIR_DEFAULT}"
    SUDO="${SUDO_DEFAULT}"
    VERSION="latest"
    INSTALL_POLKIT=0

    while [ "$#" -gt 0 ]; do
        case "$1" in
            --arch) ARCH="$2" && shift 2 ;;
            --arch=*) ARCH="${1#*=}" && shift 1 ;;
            --version) VERSION="$2" && shift 2 ;;
            --version=*) VERSION="${1#*=}" && shift 1 ;;
            --bin-dir) BIN_DIR="$2" && shift 2 ;;
            --bin-dir=*) BIN_DIR="${1#*=}" && shift 1 ;;
            --doc-dir) DOC_DIR="$2" && shift 2 ;;
            --doc-dir=*) DOC_DIR="${1#*=}" && shift 1 ;;
            --license-dir) LIC_DIR="$2" && shift 2 ;;
            --license-dir=*) LIC_DIR="${1#*=}" && shift 1 ;;
            --with-polkit) INSTALL_POLKIT=1 && shift 1 ;;
            --without-polkit) INSTALL_POLKIT=0 && shift 1 ;;
            --sudo) SUDO="$2" && shift 2 ;;
            --sudo=*) SUDO="${1#*=}" && shift 1 ;;
            -h|--help) usage && exit 0 ;;
            *) err "Unknown option: $1" ;;
        esac
    done
}

usage() {
    local _text_heading _text_reset
    _text_heading="$(tput bold || true 2>/dev/null)$(tput smul || true 2>/dev/null)"
    _text_reset="$(tput sgr0 || true 2>/dev/null)"
    cat <<EOF
Install ${PACKAGE_NAME} from https://github.com/${REPO_OWNER}/${REPO_NAME}

${_text_heading}Usage:${_text_reset}
  install.sh [OPTIONS]

${_text_heading}Options:${_text_reset}
  --arch          Override detected architecture [default: detected]
  --version       Release tag to install, without the leading 'v'
                  (e.g. 0.1.0) [default: latest]
  --bin-dir       Override installation directory [default: ${BIN_DIR_DEFAULT}]
  --doc-dir       Override documentation directory [default: ${DOC_DIR_DEFAULT}]
  --license-dir   Override license directory [default: ${LIC_DIR_DEFAULT}]
  --with-polkit   Also install the loopback helper and polkit policy under
                  /usr/libexec and /usr/share/polkit-1/actions via sudo, so
                  the GUI's LOAD VIRTUAL CAMERA button shows the branded
                  prompt
  --sudo          Override command used to elevate privileges
                  [default: ${SUDO_DEFAULT}]
  -h, --help      Print help

${_text_heading}Installs:${_text_reset}
  ${BINARIES}
EOF
}

download_package() {
    local _arch _dld _releases_url _releases _package_url _filename _package
    _arch="$1"

    if [ -n "${FRAMEIRU_INSTALL_LOCAL_PACKAGE-}" ]; then
        [ -f "${FRAMEIRU_INSTALL_LOCAL_PACKAGE}" ] || err "local package not found: ${FRAMEIRU_INSTALL_LOCAL_PACKAGE}"
        _filename=$(basename "${FRAMEIRU_INSTALL_LOCAL_PACKAGE}")
        cp -- "${FRAMEIRU_INSTALL_LOCAL_PACKAGE}" "${_filename}" || err "failed to copy local package ${FRAMEIRU_INSTALL_LOCAL_PACKAGE}"
        echo "${_filename}"
        return 0
    fi

    if check_cmd curl; then
        _dld=curl
    elif check_cmd wget; then
        _dld=wget
    else
        need_cmd 'curl or wget'
    fi

    need_cmd grep
    if [ "${VERSION}" = "latest" ]; then
        _releases_url="https://api.github.com/repos/${REPO_OWNER}/${REPO_NAME}/releases/latest"
    else
        _releases_url="https://api.github.com/repos/${REPO_OWNER}/${REPO_NAME}/releases/tags/v${VERSION}"
    fi

    case "${_dld}" in
        curl) _releases="$(curl -fsSL "${_releases_url}")" || err "curl: failed to download ${_releases_url}" ;;
        wget) _releases="$(wget -qO- "${_releases_url}")" || err "wget: failed to download ${_releases_url}" ;;
        *) err "unsupported downloader: ${_dld}" ;;
    esac

    echo "${_releases}" | grep -q 'API rate limit exceeded' &&
        err "GitHub API rate limit exceeded. Please try again later."

    _package_url="$(echo "${_releases}" | grep '"browser_download_url"' | cut -d '"' -f 4 | grep -- "${_arch}\.tar\.gz$")" ||
        err "${PACKAGE_NAME} has not been packaged for your architecture (${_arch})."

    _filename=$(basename "${_package_url}")
    _package="${_filename:-${PACKAGE_NAME}.tar.gz}"
    case "${_dld}" in
        curl) curl -fsSL -o "${_package}" "${_package_url}" || err "curl: failed to download ${_package_url}" ;;
        wget) wget -qO "${_package}" "${_package_url}" || err "wget: failed to download ${_package_url}" ;;
        *) err "unsupported downloader: ${_dld}" ;;
    esac

    CHECKSUM_URL="$(echo "${_releases}" | grep '"browser_download_url"' | cut -d '"' -f 4 | grep -- "$(basename "${_package_url}")\.sha256$" || true)"
    echo "${_package}"
}

maybe_verify_checksum() {
    local _package _checksum_file _expected _actual
    _package="$1"

    [ -n "${CHECKSUM_URL-}" ] || return 0

    _checksum_file="$(basename "${CHECKSUM_URL}")"
    if check_cmd curl; then
        curl -fsSL -o "${_checksum_file}" "${CHECKSUM_URL}" || err "curl: failed to download ${CHECKSUM_URL}"
    elif check_cmd wget; then
        wget -qO "${_checksum_file}" "${CHECKSUM_URL}" || err "wget: failed to download ${CHECKSUM_URL}"
    else
        return 0
    fi

    _expected="$(parse_checksum_file "${_checksum_file}" "$(basename "${_package}")")"
    [ -n "${_expected}" ] || return 0

    _actual="$(compute_sha256 "${_package}")"
    [ -n "${_actual}" ] || err "failed to compute SHA-256 for ${_package}"

    if [ "${_actual}" != "${_expected}" ]; then
        err "checksum mismatch for ${_package}: expected ${_expected}, got ${_actual}"
    fi

    echo "Verified SHA-256 for ${_package}"
}

parse_checksum_file() {
    local _file _target _line _sum _name
    _file="$1"
    _target="$2"
    while IFS= read -r _line; do
        [ -n "${_line}" ] || continue
        case "${_line}" in
            *" "*)
                _sum="$(printf '%s' "${_line}" | awk '{print $1}')"
                _name="$(printf '%s' "${_line}" | awk '{print $2}' | sed 's/^\*//')"
                [ "${_name}" = "${_target}" ] && { printf '%s' "${_sum}"; return 0; }
                ;;
            *)
                printf '%s' "${_line}"
                return 0
                ;;
        esac
    done <"${_file}"
    return 0
}

compute_sha256() {
    local _file
    _file="$1"
    if check_cmd sha256sum; then
        sha256sum "${_file}" | awk '{print $1}'
    elif check_cmd shasum; then
        shasum -a 256 "${_file}" | awk '{print $1}'
    elif check_cmd openssl; then
        openssl dgst -sha256 "${_file}" | awk '{print $NF}'
    else
        err "need one of: sha256sum, shasum, or openssl"
    fi
}

install_binaries() {
    local _package_root _bin
    _package_root="$1"

    ensure try_sudo mkdir -p -- "${BIN_DIR}"
    for _bin in ${BINARIES}; do
        [ -f "${_package_root}/${_bin}" ] || err "release package does not include ${_bin}"
        ensure try_sudo cp -- "${_package_root}/${_bin}" "${BIN_DIR}/${_bin}"
        ensure try_sudo chmod +x "${BIN_DIR}/${_bin}"
    done
    echo "Installed ${PACKAGE_NAME} to ${BIN_DIR}"
}

# Installs the loopback helper and polkit policy to the fixed system paths the
# policy file pins (/usr/libexec + /usr/share/polkit-1/actions). This is what
# gives the GUI's LOAD VIRTUAL CAMERA button the branded admin prompt.
install_system_loopback() {
    local _package_root _policy
    _package_root="$1"
    _policy="${_package_root}/${POLKIT_POLICY}"

    [ -f "${_package_root}/${LOOPBACK_HELPER}" ] ||
        err "release package does not include ${LOOPBACK_HELPER}"
    [ -f "${_policy}" ] || err "release package does not include ${POLKIT_POLICY}"

    ensure try_sudo mkdir -p -- /usr/libexec
    ensure try_sudo cp -- "${_package_root}/${LOOPBACK_HELPER}" /usr/libexec/frameiru-loopback-helper
    ensure try_sudo chmod 755 /usr/libexec/frameiru-loopback-helper

    ensure try_sudo mkdir -p -- /usr/share/polkit-1/actions
    ensure try_sudo cp -- "${_policy}" /usr/share/polkit-1/actions/org.frameiru.loopback.policy
    ensure try_sudo chmod 644 /usr/share/polkit-1/actions/org.frameiru.loopback.policy

    echo "Installed loopback helper to /usr/libexec and polkit policy to /usr/share/polkit-1/actions"
}

install_docs() {
    local _package_root
    _package_root="$1"

    if [ -f "${_package_root}/README.md" ]; then
        ensure try_sudo mkdir -p -- "${DOC_DIR}/${PACKAGE_NAME}"
        ensure try_sudo cp -- "${_package_root}/README.md" "${DOC_DIR}/${PACKAGE_NAME}/README.md"
        echo "Installed documentation to ${DOC_DIR}"
    fi
}

install_license() {
    local _package_root _file
    _package_root="$1"

    ensure try_sudo mkdir -p -- "${LIC_DIR}/${PACKAGE_NAME}"
    for _file in LICENSE LICENSE-APACHE; do
        if [ -f "${_package_root}/${_file}" ]; then
            ensure try_sudo cp -- "${_package_root}/${_file}" "${LIC_DIR}/${PACKAGE_NAME}/${_file}"
        fi
    done
    echo "Installed license files to ${LIC_DIR}"
}

# Non-fatal warnings: the CLI runs on a bare Linux box, but the GUI needs
# fontconfig and the virtual camera needs the v4l2loopback module.
check_runtime_deps() {
    local _have_fontconfig
    _have_fontconfig=1

    if check_cmd ldconfig; then
        if ! ldconfig -p 2>/dev/null | grep -q 'libfontconfig\.so\.1'; then
            _have_fontconfig=0
        fi
    fi
    if [ "${_have_fontconfig}" = "0" ]; then
        echo "Warning: libfontconfig not found; frameiru-ui needs it for text rendering."
        echo "  Install it with your package manager (e.g. sudo apt install libfontconfig1)."
    fi

    if [ ! -d /sys/module/v4l2loopback ]; then
        echo "Note: the v4l2loopback kernel module is not loaded."
        echo "  Run 'frameiru-cli doctor' for the exact fix (modprobe v4l2loopback ...)."
    fi
}

print_polkit_hint() {
    echo "Note: skipping system loopback install."
    echo "  The GUI's LOAD VIRTUAL CAMERA button still works (generic admin prompt),"
    echo "  but for the branded prompt rerun with --with-polkit:"
    echo "    ${PACKAGE_NAME} install.sh --with-polkit"
}

final_message() {
    echo ""
    echo "${PACKAGE_NAME} is installed!"
    if ! echo ":${PATH}:" | grep -Fq ":${BIN_DIR}:"; then
        echo "Note: ${BIN_DIR} is not on your \$PATH."
        echo "Add it to your shell profile before using ${PACKAGE_NAME}."
    fi
    echo "Preflight your setup with:"
    echo "  frameiru-cli doctor"
    echo "Start a virtual webcam with:"
    echo "  frameiru-cli run --background blur:8 --socket /tmp/frameiru.sock"
}

get_architecture() {
    local _ostype _cputype
    _ostype="$(uname -s)"
    _cputype="$(uname -m)"

    if [ "${_ostype}" != Linux ]; then
        err "Frameiru is Linux-only (virtual webcam via v4l2loopback); unsupported OS: ${_ostype}"
    fi
    if [ "$(uname -o 2>/dev/null || true)" = Android ]; then
        err "Frameiru does not support Android"
    fi

    case "${_cputype}" in
        x86_64|x86-64|x64|amd64) _cputype=x86_64 ;;
        arm64|aarch64) _cputype=aarch64 ;;
        i386|i486|i686|i786|x86|armv7*) err "unsupported CPU type: ${_cputype} (32-bit is not packaged)" ;;
        *) err "unknown CPU type: ${_cputype}" ;;
    esac

    echo "${_cputype}-unknown-linux-gnu"
}

need_cmd() {
    if ! check_cmd "$1"; then
        err "need '$1' (command not found)"
    fi
}

check_cmd() {
    command -v -- "$1" >/dev/null 2>&1
}

ensure() {
    if ! "$@"; then
        err "command failed: $*"
    fi
}

try_sudo() {
    if "$@" >/dev/null 2>&1; then
        return 0
    fi
    need_sudo
    "${SUDO}" "$@"
}

need_sudo() {
    if ! check_cmd "${SUDO}"; then
        err "could not find \`${SUDO}\`. Install sudo or rerun script with enough permissions."
    fi
    if ! "${SUDO}" -v; then
        err "sudo permissions not granted, aborting installation"
    fi
}

assert_nz() {
    if [ -z "$1" ]; then
        err "found empty string: $2"
    fi
}

err() {
    echo "Error: $1" >&2
    exit 1
}

{ main "$@" || exit 1; }