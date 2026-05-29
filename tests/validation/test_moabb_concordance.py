"""Offline + gated tests for tools/bench_moabb_concordance.py.

Mirrors the NEURAL seizure concordance test conventions
(tests/validation/test_downstream_concordance.py): load the bench module
by path, mark the offline core `l2`, and gate the heavy/real paths behind
the `data`/`slow` markers.

The hard guarantee these tests defend:
  * the bench module imports with numpy + mne + sklearn + pyedflib ONLY —
    no moabb, no codec wheel, no network;
  * a LOSSLESS codec (the real `lml` binary) is downstream-transparent
    (accuracy_delta == kappa_delta == 0.0, bit_exact True);
  * the metric is NOT trivially always-zero — a deliberately lossy
    round-trip is detectable (nonzero delta OR bit_exact False);
  * moabb stays lazy — importing the module never imports moabb, and
    load_moabb fails with a clear message when moabb is absent.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

# Offline core tier (same `l2` marker the seizure concordance uses) PLUS the
# `bci` marker so the documented `pytest -m bci` invocation (README +
# pyproject [tool.pytest.ini_options].markers) actually selects this file.
pytestmark = [pytest.mark.l2, pytest.mark.bci]

_MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "tools" / "bench_moabb_concordance.py"
)


@pytest.fixture(scope="module")
def bench():
    """Load the bench module by path (must succeed WITHOUT moabb)."""
    name = "bench_moabb_concordance_under_test"
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Module-import gating: moabb must never be imported by importing the module.
# ---------------------------------------------------------------------------
def test_module_imports_without_moabb(bench):
    # This env has no moabb at all; importing the bench must not have
    # pulled it in (and must not be required to).
    assert "moabb" not in sys.modules
    # The lazy entry point exists but is not invoked at import time.
    assert hasattr(bench, "load_moabb")


# ---------------------------------------------------------------------------
# The pipeline + dataset are real (not a rigged 50/50).
# ---------------------------------------------------------------------------
def test_synthetic_is_decodable(bench):
    X, y, sfreq, ch_names = bench.synthetic_bci_dataset(seed=0)
    # Balanced two-class.
    assert set(np.unique(y)) == {0, 1}
    assert np.sum(y == 0) == np.sum(y == 1)
    score = bench.decoding_score(X, y, sfreq, seed=0)
    # Clearly above chance — proves CSP+LDA actually separates the classes.
    assert score["accuracy"] >= 0.75, score
    assert score["kappa"] > 0.0


def test_synthetic_shapes(bench):
    X, y, sfreq, ch_names = bench.synthetic_bci_dataset(
        n_trials=20, n_ch=6, n_times=128, sfreq=100.0, seed=3
    )
    assert X.shape == (20, 6, 128)
    assert y.shape == (20,)
    assert sfreq == 100.0
    assert ch_names == [f"EEG{i}" for i in range(6)]
    assert X.dtype == np.float64


# ---------------------------------------------------------------------------
# quantize_to_int16 — range + determinism + round-trip preservation.
# ---------------------------------------------------------------------------
def test_quantize_range_and_dtype(bench):
    rng = np.random.default_rng(1)
    X = rng.standard_normal((10, 4, 64)) * 7.3
    Xi = bench.quantize_to_int16(X)
    assert Xi.dtype == np.int16
    assert Xi.min() >= -32768
    assert Xi.max() <= 32767
    # The global peak maps near the +/-32000 headroom target.
    assert np.max(np.abs(Xi)) >= 31000


def test_quantize_is_deterministic(bench):
    rng = np.random.default_rng(2)
    X = rng.standard_normal((5, 3, 32))
    a = bench.quantize_to_int16(X)
    b = bench.quantize_to_int16(X)
    assert np.array_equal(a, b)


def test_quantize_zero_signal_safe(bench):
    X = np.zeros((4, 2, 16))
    Xi = bench.quantize_to_int16(X)
    assert Xi.dtype == np.int16
    assert np.array_equal(Xi, np.zeros_like(Xi))


# ---------------------------------------------------------------------------
# resolve_lml_bin — resolution order ($LML_BIN first).
# ---------------------------------------------------------------------------
def test_resolve_lml_bin_env_first(bench, tmp_path, monkeypatch):
    fake = tmp_path / "lml"
    fake.write_text("#!/bin/sh\n")
    monkeypatch.setenv("LML_BIN", str(fake))
    resolved = bench.resolve_lml_bin()
    assert resolved == str(fake.resolve())


def test_resolve_lml_bin_env_missing_file_ignored(bench, tmp_path, monkeypatch):
    # $LML_BIN pointing at a non-file must NOT win; falls through.
    monkeypatch.setenv("LML_BIN", str(tmp_path / "does-not-exist"))
    monkeypatch.setattr(bench.shutil, "which", lambda _name: None)
    # Force the sibling path to not exist by pointing REPO_ROOT elsewhere.
    monkeypatch.setattr(bench, "REPO_ROOT", tmp_path / "isolated" / "Eagle")
    assert bench.resolve_lml_bin() is None


def test_resolve_lml_bin_path_fallback(bench, tmp_path, monkeypatch):
    monkeypatch.delenv("LML_BIN", raising=False)
    monkeypatch.setattr(bench, "REPO_ROOT", tmp_path / "isolated" / "Eagle")
    monkeypatch.setattr(bench.shutil, "which", lambda name: "/usr/bin/lml" if name == "lml" else None)
    assert bench.resolve_lml_bin() == "/usr/bin/lml"


# ---------------------------------------------------------------------------
# load_moabb stays lazy + fails clearly when moabb is absent.
# ---------------------------------------------------------------------------
def test_load_moabb_gated(bench, monkeypatch):
    # Importing the module did not import moabb.
    assert "moabb" not in sys.modules
    # Simulate moabb being absent: block its import.
    import builtins

    real_import = builtins.__import__

    def _blocked(name, *a, **k):
        if name == "moabb" or name.startswith("moabb."):
            raise ImportError("No module named 'moabb'")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    with pytest.raises(ImportError) as exc:
        bench.load_moabb()
    # The error is actionable: it names the install path.
    assert "moabb" in str(exc.value)
    assert "pip install" in str(exc.value)


def test_load_moabb_rejects_unsupported_pair(bench, monkeypatch):
    # When moabb IS importable, an unsupported dataset/paradigm raises
    # ValueError before any get_data() call. Stub a minimal moabb.
    import types

    fake_moabb = types.ModuleType("moabb")
    fake_datasets = types.ModuleType("moabb.datasets")
    fake_paradigms = types.ModuleType("moabb.paradigms")
    monkeypatch.setitem(sys.modules, "moabb", fake_moabb)
    monkeypatch.setitem(sys.modules, "moabb.datasets", fake_datasets)
    monkeypatch.setitem(sys.modules, "moabb.paradigms", fake_paradigms)
    with pytest.raises(ValueError):
        bench.load_moabb(dataset="NotARealDataset")


# ---------------------------------------------------------------------------
# Lossy round-trip IS detected — guards against a self-fooling always-0 bench.
# ---------------------------------------------------------------------------
def test_lossy_roundtrip_is_detected(bench):
    X, y, sfreq, ch_names = bench.synthetic_bci_dataset(seed=0)

    def lossy_roundtrip(X_int, sfreq, ch_names):
        # A deliberately destructive "codec": replace every sample with
        # deterministic noise, annihilating the class-discriminative spatial
        # structure. This drives the recon decoder to chance AND breaks
        # bit-exactness — both axes of the metric must register it.
        rng = np.random.default_rng(123)
        return rng.integers(-2000, 2000, size=X_int.shape, dtype=np.int64).astype(np.int16)

    result = bench.concordance(
        X, y, sfreq, ch_names, lml_bin=None, seed=0, roundtrip_fn=lossy_roundtrip
    )
    # The metric must be able to FIRE on BOTH axes: the samples differ
    # (not bit-exact) AND the decoding accuracy genuinely degrades. This is
    # the guard against a self-fooling always-0 bench — if the harness
    # silently always returned Δ=0 / bit_exact=True, this destructive codec
    # would (wrongly) sail through.
    assert result["bit_exact"] is False
    assert result["accuracy_delta"] != 0.0
    assert result["transparent"] is False
    # Recon should collapse toward chance for a 2-class problem.
    assert result["accuracy_recon"] < result["accuracy_orig"]


def test_identity_roundtrip_is_zero_delta(bench):
    # Sanity: an identity codec (perfect lossless) gives exactly 0 delta
    # even without the lml binary — confirms the comparison plumbing.
    X, y, sfreq, ch_names = bench.synthetic_bci_dataset(seed=0)

    def identity_roundtrip(X_int, sfreq, ch_names):
        # Round-trip through the SAME EDF write/read the reference uses.
        return bench._read_edf_reference(X_int, sfreq, ch_names)

    result = bench.concordance(
        X, y, sfreq, ch_names, lml_bin=None, seed=0, roundtrip_fn=identity_roundtrip
    )
    assert result["bit_exact"] is True
    assert result["accuracy_delta"] == 0.0
    assert result["kappa_delta"] == 0.0


# ---------------------------------------------------------------------------
# The headline: a real LOSSLESS codec is downstream-transparent (Δ == 0).
# Skipped when no lml binary is resolvable.
# ---------------------------------------------------------------------------
def test_lossless_concordance_zero_delta(bench):
    lml_bin = bench.resolve_lml_bin()
    if lml_bin is None:
        pytest.skip("no lml binary (set $LML_BIN or build LamQuant-Lossless)")

    X, y, sfreq, ch_names = bench.synthetic_bci_dataset(seed=0)
    result = bench.concordance(X, y, sfreq, ch_names, lml_bin=lml_bin, seed=0)

    # Bit-exact round-trip ⇒ identical samples ⇒ identical decoding.
    assert result["bit_exact"] is True, result
    assert result["accuracy_delta"] == 0.0, result
    assert result["kappa_delta"] == 0.0, result
    # And the reference itself is genuinely decodable (not a 50/50 fluke).
    assert result["accuracy_orig"] >= 0.75, result
    assert result["transparent"] is True


def test_codec_roundtrip_is_bit_exact(bench):
    """codec_roundtrip_edf returns samples identical to the EDF reference."""
    lml_bin = bench.resolve_lml_bin()
    if lml_bin is None:
        pytest.skip("no lml binary (set $LML_BIN or build LamQuant-Lossless)")

    X, y, sfreq, ch_names = bench.synthetic_bci_dataset(n_trials=8, n_ch=4, seed=1)
    X_int = bench.quantize_to_int16(X)
    Xq = bench._read_edf_reference(X_int, sfreq, ch_names)
    Xr = bench.codec_roundtrip_edf(X_int, sfreq, ch_names, lml_bin)
    assert Xr.shape == Xq.shape == X_int.shape
    assert np.array_equal(Xq, Xr)


# ---------------------------------------------------------------------------
# Real-data path: needs the moabb extra + a corpus. Gated, never run offline.
# ---------------------------------------------------------------------------
@pytest.mark.data
@pytest.mark.slow
def test_moabb_real_concordance():  # pragma: no cover - requires moabb + data
    pytest.importorskip("moabb", reason="needs the moabb extra: pip install -e '.[moabb]'")
    spec = importlib.util.spec_from_file_location("bench_moabb_concordance_real", _MODULE_PATH)
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)

    lml_bin = bench.resolve_lml_bin()
    if lml_bin is None:
        pytest.skip("no lml binary")

    X, y, sfreq, ch_names = bench.load_moabb(
        dataset="BNCI2014_001", paradigm="LeftRightImagery", subject=1
    )
    result = bench.concordance(X, y, sfreq, ch_names, lml_bin=lml_bin, seed=0)
    assert result["bit_exact"] is True
    assert result["accuracy_delta"] == 0.0
    assert result["kappa_delta"] == 0.0


# ---------------------------------------------------------------------------
# CLI smoke (offline synthetic path through the real binary).
# ---------------------------------------------------------------------------
def test_main_synthetic(bench, tmp_path):
    lml_bin = bench.resolve_lml_bin()
    if lml_bin is None:
        pytest.skip("no lml binary")
    out = tmp_path / "result.json"
    rc = bench.main(["--source", "synthetic", "--lml-bin", lml_bin, "--out", str(out)])
    assert rc == 0
    import json

    payload = json.loads(out.read_text())
    assert payload["bit_exact"] is True
    assert payload["accuracy_delta"] == 0.0


def test_main_moabb_absent_clean_exit(bench, monkeypatch, capsys):
    """--source moabb with moabb absent → clean message + nonzero, no traceback."""
    lml_bin = bench.resolve_lml_bin()
    if lml_bin is None:
        # main() returns 2 before ever touching moabb when no binary.
        rc = bench.main(["--source", "moabb"])
        assert rc == 2
        return
    import builtins

    real_import = builtins.__import__

    def _blocked(name, *a, **k):
        if name == "moabb" or name.startswith("moabb."):
            raise ImportError("No module named 'moabb'")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    rc = bench.main(["--source", "moabb", "--lml-bin", lml_bin])
    assert rc != 0
    err = capsys.readouterr().err
    assert "moabb" in err
    assert "Traceback" not in err
