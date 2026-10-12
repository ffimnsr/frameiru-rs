#!/bin/sh
# shellcheck shell=dash
# Shared githooks helpers: test-command canonicalization, the state-gated
# test cache, and the .githooks -> hooks-dir drift check. Sourced by
# .githooks/pre-commit and .githooks/record-tests-pass; functions only,
# nothing runs on source.
#
# State design: the full workspace test suite is expensive, so it only runs
# when the testable source tree changed since the last recorded pass. The
# state is a content hash over everything a workspace test can depend on
# (tracked plus non-ignored untracked files under `crates/`, the workspace
# manifest, and Cargo.lock), paired with the exact canonical test command.
# README/docs edits do not invalidate the state, and "staged but not yet
# committed" never matters: only file contents do.

# The one source of truth for what "full workspace test" means. The
# pre-commit hook runs exactly this when the state is stale; `mk ci` runs it
# and then records a pass via .githooks/record-tests-pass, so tests run once
# per tree change no matter who ran them first.
githooks_test_cmd() {
    printf '%s\n' 'cargo test --quiet --workspace --locked'
}

# State file lives inside target/ (gitignored): `cargo clean` wipes it too,
# forcing a fresh test run.
githooks_state_file() {
    if [ -n "${FRAMEIRU_PRE_COMMIT_STATE_FILE:-}" ]; then
        printf '%s\n' "${FRAMEIRU_PRE_COMMIT_STATE_FILE}"
        return 0
    fi
    printf '%s\n' "$(git rev-parse --show-toplevel)/target/.githooks-state"
}

# Content hash of the test scope: every tracked path under crates/ plus the
# workspace manifest and Cargo.lock, then non-ignored untracked files in the
# same scope (new modules). Output is "<path> <sha256>" lines piped into the
# final digest, so renames and deletions also change the key.
githooks_key_scope() {
    {
        git ls-files -z -- crates Cargo.toml Cargo.lock
        git ls-files -o --exclude-standard -z -- crates Cargo.toml Cargo.lock
    } | sort -z -u | tr '\0' '\n' | while IFS= read -r _f; do
        [ -f "${_f}" ] || continue
        printf '%s ' "${_f}"
        githooks_sha256_file "${_f}"
    done
}

githooks_compute_key() {
    githooks_key_scope | githooks_sha256_stdin
}

githooks_sha256_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1"
    else
        shasum -a 256 "$1"
    fi | cut -d' ' -f1
}

githooks_sha256_stdin() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum
    else
        shasum -a 256
    fi | cut -d' ' -f1
}

# Exit 0 when the state file records a pass of exactly $1 with key $2.
githooks_state_matches() {
    local _cmd _key _file
    _cmd="$1"
    _key="$2"
    _file="$(githooks_state_file)"
    [ -f "${_file}" ] || return 1
    [ "$(sed -n 's/^command=//p' "${_file}")" = "${_cmd}" ] || return 1
    [ "$(sed -n 's/^key=//p' "${_file}")" = "${_key}" ] || return 1
}

# Atomically record a pass of $1 (command) with key $2.
githooks_record_pass() {
    local _cmd _key _file _dir _tmp
    _cmd="$1"
    _key="$2"
    _file="$(githooks_state_file)"
    _dir="$(dirname "${_file}")"
    mkdir -p "${_dir}"
    _tmp="${_file}.tmp.$$"
    {
        printf 'command=%s\n' "${_cmd}"
        printf 'key=%s\n' "${_key}"
        printf 'when=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u)"
    } >"${_tmp}"
    mv -f "${_tmp}" "${_file}"
}

# Prints the recorded pass timestamp; exit 1 when there is no state.
githooks_state_when() {
    local _file
    _file="$(githooks_state_file)"
    [ -f "${_file}" ] || return 1
    sed -n 's/^when=//p' "${_file}"
}

# Hook files managed by this repo: tracked plus non-ignored untracked under
# .githooks/ (a freshly added hook file is drift-checked before it lands in a
# commit). Prints repo-root-relative paths, one per line.
githooks_managed_files() {
    {
        git ls-files -z -- .githooks
        git ls-files -o --exclude-standard -z -- .githooks
    } | sort -z -u | tr '\0' '\n'
}

# Canonicalize a path that may not exist yet (pwd -P resolves symlinks so a
# core.hooksPath symlink to .githooks compares equal to .githooks itself).
githooks_abspath() {
    (CDPATH='' cd -- "$1" 2>/dev/null && pwd -P) || printf '%s\n' "$1"
}

# Warn (never fail) when the installed hook files are not byte-identical to
# their .githooks/ sources. install-githooks copies files instead of linking
# them, so a pull or an edit leaves the installed copy running silently stale
# until the task is re-run. Prints one line per drifted file plus the refresh
# command; always exits 0. No-op when the hooks dir *is* .githooks (in-place
# setup via core.hooksPath) or outside a git repository.
githooks_check_drift() {
    local _root _src _dst _display _rel _name _warned
    _root="$(git rev-parse --show-toplevel 2>/dev/null)" || return 0
    _src="${_root}/.githooks"
    # Resolve from the root: --git-path output is relative to $PWD (e.g.
    # ../.git/hooks in a subdirectory) and stays root-relative this way.
    _dst="$(CDPATH='' cd -- "${_root}" && git rev-parse --git-path hooks 2>/dev/null)" || return 0
    case "${_dst}" in
    /*) ;;
    *) _dst="${_root}/${_dst}" ;;
    esac
    if [ "$(githooks_abspath "${_src}")" = "$(githooks_abspath "${_dst}")" ]; then
        return 0
    fi
    _display="${_dst#"${_root}/"}"
    _warned=0
    # Heredoc, not a pipe: the loop body sets _warned, which a piped subshell
    # would drop.
    while IFS= read -r _rel; do
        [ -n "${_rel}" ] || continue
        _name="${_rel#.githooks/}"
        [ -f "${_src}/${_name}" ] || continue
        if [ ! -e "${_dst}/${_name}" ]; then
            printf '[githooks] drift: .githooks/%s is not installed under %s\n' "${_name}" "${_display}"
            _warned=1
        elif ! cmp -s "${_src}/${_name}" "${_dst}/${_name}"; then
            printf '[githooks] drift: %s/%s differs from .githooks/%s\n' "${_display}" "${_name}" "${_name}"
            _warned=1
        fi
    done <<EOF
$(githooks_managed_files)
EOF
    for _name in pre-commit record-tests-pass; do
        if [ -e "${_src}/${_name}" ] && [ -e "${_dst}/${_name}" ]; then
            # Nested: a bare `A && B && C` chain would trip `set -e` when the
            # hook is installed correctly and the last test is false.
            if [ ! -x "${_dst}/${_name}" ]; then
                printf '[githooks] drift: %s/%s is not executable; git skips hooks without the exec bit\n' "${_display}" "${_name}"
                _warned=1
            fi
        fi
    done
    # Leftovers: installed files this repo no longer ships. Only names the
    # hook owns are checked, so unrelated user hooks (commit-msg, ...) stay
    # out of it.
    for _name in pre-commit record-tests-pass; do
        if [ -e "${_dst}/${_name}" ] && [ ! -e "${_src}/${_name}" ]; then
            printf '[githooks] drift: %s/%s is installed but no longer in .githooks/\n' "${_display}" "${_name}"
            _warned=1
        fi
    done
    for _rel in "${_dst}"/lib/*; do
        [ -e "${_rel}" ] || continue
        _name="lib/${_rel##*/}"
        if [ ! -e "${_src}/${_name}" ]; then
            printf '[githooks] drift: %s/%s is installed but no longer in .githooks/\n' "${_display}" "${_name}"
            _warned=1
        fi
    done
    if [ "${_warned}" -eq 1 ]; then
        printf '[githooks] warning: installed hooks are stale; run mk install-githooks to refresh\n'
    fi
    return 0
} 