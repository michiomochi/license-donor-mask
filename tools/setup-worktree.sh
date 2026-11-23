#!/bin/sh
# Set up this worktree's Python .venv from uv.lock, offline from the uv cache first.
#
#   sh tools/setup-worktree.sh
#
# Each worktree gets its own .venv (not shared with the primary checkout). "Ready" means .venv/pyvenv.cfg exists and the
# uv.lock the venv was built from (.venv/.setup-worktree-uv.lock) equals the current uv.lock; otherwise it runs
# `uv sync --frozen --offline`, and online when the cache lacks something. It never rewrites uv.lock.
#
# Exit codes: 0 = already set up or set up now / 1 = failure (or uv is missing).
set -eu

root=$(git rev-parse --show-toplevel)
cd "$root"

uv_run() {
  if command -v uv >/dev/null 2>&1; then
    uv "$@"
  elif command -v mise >/dev/null 2>&1; then
    mise exec -- uv "$@"
  else
    return 127
  fi
}

if ! uv_run --version >/dev/null 2>&1; then
  echo "uv is missing, so .venv is not created (install uv with mise)" >&2
  exit 1
fi

ready() {
  [ -f .venv/pyvenv.cfg ] && [ -f .venv/.setup-worktree-uv.lock ] && cmp -s uv.lock .venv/.setup-worktree-uv.lock
}

if ready; then
  echo "Dependencies are ready"
  exit 0
fi

log=$(mktemp)
trap 'rm -f "$log"' EXIT
start=$(date +%s)
if ! uv_run sync --frozen --offline >"$log" 2>&1; then
  if ! uv_run sync --frozen >"$log" 2>&1; then
    cat "$log" >&2
    echo "uv sync failed" >&2
    exit 1
  fi
fi
cp uv.lock .venv/.setup-worktree-uv.lock
echo "Set up .venv from uv.lock ($(($(date +%s) - start)) seconds)"
