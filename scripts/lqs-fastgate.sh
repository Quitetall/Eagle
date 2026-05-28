#!/usr/bin/env bash
# lqs-fastgate.sh — local fast gate for the LQS Rust crate.
#
# The fast local mirror of the CI `lqs-rust` job: build + test the `lqs`
# crate, print PASS/FAIL with timing, exit nonzero on failure. Meant for
# pre-commit / pre-push so you get codec-correctness feedback in seconds
# instead of waiting on CI (the whole reason LQS is Rust-canonical:
# faster CI / pre-commit than the Python path).
#
# Scope intentionally narrow — build + test only. No fmt (advisory in CI),
# no --all-features (the optional `zstd` feature needs a system dep we
# skip to stay fast). The smoke `store` run is covered by `cargo test`'s
# lossless-grading cases; pre-push stays lean.
#
# Enable as a hook:
#   chmod +x scripts/lqs-fastgate.sh .githooks/pre-push
#   git config core.hooksPath .githooks
#
# Or run by hand from anywhere in the repo:
#   ./scripts/lqs-fastgate.sh

set -euo pipefail

# Resolve repo root from this script's location so it works regardless of
# the caller's cwd (hooks run from the repo root, but humans may not).
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# Per the LQS gate contract: operate from the crate dir. `lqs` is a member
# of the repo-root workspace, so cargo walks up to the root Cargo.lock /
# target automatically — the build dir is shared with CI's cache.
cd "${REPO_ROOT}/lqs"

start="$(date +%s)"

fail() {
  local elapsed=$(( $(date +%s) - start ))
  echo ""
  echo "lqs-fastgate: FAIL ($1) — ${elapsed}s"
  exit 1
}

echo "lqs-fastgate: building lqs ..."
cargo build -q || fail "build"

echo "lqs-fastgate: testing lqs ..."
cargo test -q || fail "test"

elapsed=$(( $(date +%s) - start ))
echo ""
echo "lqs-fastgate: PASS — ${elapsed}s"
