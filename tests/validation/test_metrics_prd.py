"""tests/test_metrics_prd.py — PRD + per-band PRD + LQS gating contracts.

R alone doesn't tell the full story: a reconstruction can have R=0.95
(shape) and PRD=25% (wrong amplitude). PRD is the co-equal primary
metric. These tests pin down the math + the LQS ship/no-ship gate.
"""
from __future__ import annotations

import pytest  # decomp(eagle): skip when sibling LamQuant-Neural / -Lossless not installed
pytest.importorskip("metrics", reason="Cross-repo test; needs sibling LamQuant-Neural or -Lossless wheel installed")

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / 'ai_models'))

from metrics import (
    EEG_BANDS,
    prd_numpy, prd_torch, pearson_r_numpy,
    per_band_prd, per_band_r,
    lqs_compliance, lqs_pretty,
)


# ============================================================
# PRD math — closed-form sanity checks
# ============================================================

def test_prd_zero_when_perfect_reconstruction():
    sig = np.random.RandomState(0).randn(10, 21, 2500)
    assert prd_numpy(sig, sig) == pytest.approx(0.0, abs=1e-9)


def test_prd_100_when_signal_zeroed_out():
    """Reconstruction = 0 → noise = signal → PRD = 100%."""
    sig = np.random.RandomState(1).randn(2500)
    recon = np.zeros_like(sig)
    assert prd_numpy(sig, recon) == pytest.approx(100.0, rel=1e-6)


def test_prd_handles_zero_signal():
    """All-zero original + matching all-zero recon → PRD = 0 (no NaN)."""
    z = np.zeros(1000)
    assert prd_numpy(z, z) == 0.0


def test_prd_torch_matches_numpy():
    import torch
    rng = np.random.RandomState(7)
    sig = rng.randn(4, 21, 2500).astype(np.float32)
    recon = sig + 0.1 * rng.randn(*sig.shape).astype(np.float32)
    p_np = prd_numpy(sig, recon)
    p_torch = float(prd_torch(torch.from_numpy(sig), torch.from_numpy(recon)))
    assert p_np == pytest.approx(p_torch, rel=1e-3), (
        f'torch and numpy PRD diverged: {p_np} vs {p_torch}'
    )


def test_prd_torch_is_differentiable():
    import torch
    target = torch.randn(2, 21, 2500)
    recon = torch.zeros_like(target, requires_grad=True)
    loss = prd_torch(target, recon)
    loss.backward()
    assert recon.grad is not None and recon.grad.abs().sum() > 0, (
        'prd_torch did not produce a gradient — not usable as training loss'
    )


# ============================================================
# Per-band PRD — bandpass localization
# ============================================================

def test_per_band_prd_localizes_band_specific_error():
    """Inject error only in the alpha band → alpha PRD spikes; others stay low."""
    fs = 250.0; T = 2500
    t = np.arange(T) / fs
    # Broadband signal: power in all five bands
    sig = sum(np.sin(2 * np.pi * f * t) for f in (2, 6, 10, 20, 35))
    # Reconstruction missing alpha-band component (8-13 Hz)
    recon = sig - np.sin(2 * np.pi * 10 * t)
    pb = per_band_prd(sig, recon, fs=fs)
    assert pb['alpha'] > 50, (
        f'alpha-band-only error not localized: {pb}'
    )
    # Other bands should not spike
    for b in ('delta', 'theta', 'beta', 'gamma'):
        assert pb[b] < pb['alpha'], (
            f'{b} PRD ({pb[b]:.1f}) >= alpha PRD ({pb["alpha"]:.1f}) — '
            f'bandpass leakage'
        )


def test_per_band_prd_bands_cover_the_canonical_eeg_frequencies():
    """The five bands must cover delta through gamma without gap or overlap."""
    bands = list(EEG_BANDS.values())
    bands.sort()
    for i in range(len(bands) - 1):
        assert bands[i][1] == bands[i + 1][0], (
            f'gap or overlap between bands at {bands[i][1]} Hz'
        )


# ============================================================
# LQS gating — the ship/no-ship gate
# ============================================================

def test_lqs_returns_clinical_when_all_thresholds_met():
    """R=0.97, PRD=4%, all per-bands well below LQS-C limits → LQS-C."""
    level, viol = lqs_compliance(
        val_r=0.97, val_prd=4.0,
        per_band_prd_dict={'delta': 2.0, 'theta': 3.0, 'alpha': 4.0,
                           'beta': 6.0, 'gamma': 10.0},
        per_band_r_dict={'delta': 0.99, 'theta': 0.98, 'alpha': 0.97,
                         'beta': 0.95, 'gamma': 0.90},
    )
    assert level == 'C', f'expected C, got {level!r} (violations: {viol})'
    assert viol == []


def test_lqs_drops_to_monitoring_when_beta_prd_too_high():
    """Reproduces the user's example: R=0.8981, PRD=11.2%, beta/gamma over."""
    level, viol = lqs_compliance(
        val_r=0.8981, val_prd=11.2,
        per_band_prd_dict={'delta': 3.7, 'theta': 5.2, 'alpha': 5.6,
                           'beta': 12.9, 'gamma': 25.8},
    )
    assert level == 'M', f'expected M, got {level!r}'
    # The blocking violations should call out global R, global PRD, beta, gamma.
    joined = '\n'.join(viol)
    assert 'global R' in joined
    assert 'global PRD' in joined
    assert 'beta PRD' in joined
    assert 'gamma PRD' in joined


def test_lqs_returns_alerting_at_borderline_quality():
    """R=0.75, PRD=30% → LQS-A."""
    level, _ = lqs_compliance(val_r=0.75, val_prd=30.0,
                               per_band_prd_dict={'delta': 18, 'theta': 22,
                                                   'alpha': 28, 'beta': 35,
                                                   'gamma': 50})
    assert level == 'A', f'expected A, got {level!r}'


def test_lqs_returns_empty_when_below_alerting_floor():
    """R=0.5 (below LQS-A min_r=0.7) → not deployable at any tier."""
    level, viol = lqs_compliance(val_r=0.5, val_prd=80.0)
    assert level == ''
    assert any('global R' in v for v in viol), (
        f'expected R violation in {viol}'
    )


def test_lqs_pretty_renders_all_five_bands():
    """The dashboard line must show all five bands with greek glyphs."""
    s = lqs_pretty(0.0, 0.0,
                    {'delta': 1.2, 'theta': 2.3, 'alpha': 3.4,
                     'beta': 4.5, 'gamma': 5.6})
    for glyph in ('δ', 'θ', 'α', 'β', 'γ'):
        assert glyph in s, f'missing {glyph} in {s!r}'


# ============================================================
# EpochReport / RunSummary schema — ship/no-ship fields exist
# ============================================================

def test_epoch_report_has_prd_fields():
    """The dataclass must have val_prd + per-band PRD fields."""
    sys.path.insert(0, str(_REPO / 'ai_models'))
    from training_types import EpochReport
    r = EpochReport()
    for f in ('val_prd', 'best_val_prd',
              'val_prd_delta', 'val_prd_theta', 'val_prd_alpha',
              'val_prd_beta', 'val_prd_gamma'):
        assert hasattr(r, f), f'EpochReport missing {f}'


def test_run_summary_has_lqs_fields():
    sys.path.insert(0, str(_REPO / 'ai_models'))
    from training_types import RunSummary
    s = RunSummary()
    for f in ('best_val_prd', 'final_val_prd', 'lqs_level', 'lqs_violations',
              'best_prd_delta', 'best_prd_theta', 'best_prd_alpha',
              'best_prd_beta', 'best_prd_gamma'):
        assert hasattr(s, f), f'RunSummary missing {f}'
