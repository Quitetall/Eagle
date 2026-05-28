"""Run `scripts/audit_format_consistency.py` as a pytest gate.

Integrated 2026-05-18 (cleanup pass — auditors that belong in the
test suite shouldn't bloat `scripts/`). The audit script itself
remains callable from the command line; this test just wraps it so
the CI gate runs uniformly with the rest of `pytest tests/`.

The audit checks that magic bytes, header sizes, struct formats,
and constants stay consistent across Python (`lamquant_codec`),
Rust (`lamquant-core`), docs, and tests. Any drift fires
`sys.exit(1)`; the wrapper turns that into a pytest failure.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    """Walk upward looking for ``pyproject.toml``."""
    here = Path(__file__).resolve().parent
    for ancestor in [here, *here.parents]:
        if (ancestor / "pyproject.toml").exists():
            return ancestor
    raise RuntimeError("can't find repo root from test file")


def test_format_consistency():
    """The cross-language constant audit must pass."""
    script = _repo_root() / "scripts" / "audit_format_consistency.py"
    # decomp(eagle): audit script is monorepo tooling (scripts/ split
    # per-product during the 8-repo decomposition). Skip cleanly when
    # running Eagle standalone without it present.
    if not script.exists():
        import pytest
        pytest.skip(f"audit script not in this repo (monorepo-only tooling): {script}")
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        cwd=_repo_root(),
    )
    if result.returncode != 0:
        raise AssertionError(
            "audit_format_consistency reported drift between Python / "
            "Rust / docs / tests:\n\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )
