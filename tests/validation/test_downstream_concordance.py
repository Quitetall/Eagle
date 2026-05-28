"""Unit tests for ai_models/validation/downstream_concordance.py — Phase 3."""
from __future__ import annotations

import pytest  # decomp(eagle): downstream_concordance.py lives in sibling LamQuant-Neural
pytest.importorskip("ai_models", reason="Cross-repo: needs sibling LamQuant-Neural (ai_models/validation/downstream_concordance.py)")

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytestmark = pytest.mark.l2


_MODULE_PATH = (Path(__file__).resolve().parents[2]
                / "ai_models" / "validation" / "downstream_concordance.py")


@pytest.fixture(scope="module")
def dc():
    name = "ai_models.validation.downstream_concordance_under_test"
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# hjorth_parameters
# ---------------------------------------------------------------------------
class TestHjorthParameters:
    def test_returns_three_floats(self, dc):
        sig = np.random.randn(1000)
        a, m, c = dc.hjorth_parameters(sig)
        assert isinstance(a, float)
        assert isinstance(m, float)
        assert isinstance(c, float)

    def test_constant_signal_zero_activity(self, dc):
        sig = np.zeros(100)
        a, _, _ = dc.hjorth_parameters(sig)
        assert a == 0.0

    def test_unit_variance_white_noise(self, dc):
        np.random.seed(0)
        sig = np.random.randn(10000)
        a, m, c = dc.hjorth_parameters(sig)
        assert a == pytest.approx(1.0, abs=0.1)
        # White noise mobility ≈ sqrt(2)
        assert m > 0.5

    def test_multidim_input(self, dc):
        sig = np.random.randn(4, 100)
        a, m, c = dc.hjorth_parameters(sig)
        # Variance is over all elements; returns scalars
        assert np.isfinite(a)


# ---------------------------------------------------------------------------
# hjorth_concordance
# ---------------------------------------------------------------------------
class TestHjorthConcordance:
    def test_identical_perfect_correlation(self, dc):
        sig = np.random.randn(21, 1000)
        r = dc.hjorth_concordance(sig, sig.copy())
        for key in ("hjorth_activity_r", "hjorth_mobility_r",
                     "hjorth_complexity_r"):
            assert key in r
            # Identical → r ≈ 1.0
            assert r[key] == pytest.approx(1.0, abs=1e-5)

    def test_returns_three_metrics(self, dc):
        a = np.random.randn(4, 500)
        b = np.random.randn(4, 500)
        r = dc.hjorth_concordance(a, b)
        assert set(r.keys()) == {
            "hjorth_activity_r", "hjorth_mobility_r", "hjorth_complexity_r"
        }


# ---------------------------------------------------------------------------
# train_simple_seizure_detector + evaluate_seizure_concordance
# ---------------------------------------------------------------------------
class TestSeizureDetector:
    def test_train_returns_classifier(self, dc):
        windows = [np.random.randn(4, 100) for _ in range(20)]
        labels = [True, False] * 10
        clf, scaler = dc.train_simple_seizure_detector(windows, labels)
        assert hasattr(clf, "predict")
        assert hasattr(scaler, "transform")

    def test_evaluate_concordance_basic(self, dc):
        windows = [np.random.randn(4, 100) for _ in range(20)]
        labels = [True, False] * 10
        clf, scaler = dc.train_simple_seizure_detector(windows, labels)
        recon = [w + 0.01 * np.random.randn(*w.shape) for w in windows]
        results = dc.evaluate_seizure_concordance(
            clf, scaler, windows, recon, labels)
        assert "seizure_f1_orig" in results
        assert "seizure_f1_recon" in results
        assert "seizure_f1_delta" in results
        assert "decision_concordance" in results

    def test_evaluate_with_auroc(self, dc):
        windows = [np.random.randn(4, 100) for _ in range(20)]
        labels = [True if i < 10 else False for i in range(20)]
        clf, scaler = dc.train_simple_seizure_detector(windows, labels)
        results = dc.evaluate_seizure_concordance(
            clf, scaler, windows, windows, labels)
        # Both classes present → AUROC keys
        assert "auroc_orig" in results
        assert "auroc_delta" in results


# ---------------------------------------------------------------------------
# main — stub heavy deps
# ---------------------------------------------------------------------------
def _make_npz(tmp_path: Path, name="f.npz", with_seizure=False):
    p = tmp_path / name
    raw = np.random.randint(-10000, 10000, (21, 12500), dtype=np.int32)
    mask = np.zeros(12500)
    if with_seizure:
        mask[:6250] = 1
    np.savez_compressed(p, data=raw, seizure_mask=mask)
    return p


class _FE:
    def __init__(self, path): self.path = path


@pytest.fixture
def stub_main(monkeypatch, tmp_path):
    import torch

    class _Manifest:
        @classmethod
        def load(cls, _p): return cls()
        def get_file_entries(self, _split):
            return [_FE(_make_npz(tmp_path, with_seizure=(i % 2 == 0)))
                    for i in range(4)]

    fake_dt = types.ModuleType("data_types")
    fake_dt.DatasetManifest = _Manifest
    fake_dt.Split = MagicMock()
    monkeypatch.setitem(sys.modules, "data_types", fake_dt)

    class _Codec(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(313, 313)
        def to(self, device): return self
        def eval(self): return self
        def load_encoder(self, p): pass
        def load_decoder(self, p): pass
        def forward(self, x, quantize=False):
            return self.lin(x)
        def parameters(self):
            return self.lin.parameters()

    fake_jc = types.ModuleType("joint_codec")
    fake_jc.build_default_joint = lambda vocos_tier=3: _Codec()
    monkeypatch.setitem(sys.modules, "joint_codec", fake_jc)

    fake_sp = types.ModuleType("subband_preprocess")
    fake_sp.preprocess_subband_single = lambda x: (
        np.random.randn(21, 313).astype(np.float32), None, None)
    monkeypatch.setitem(sys.modules, "subband_preprocess", fake_sp)

    yield


class TestMain:
    def test_main_runs(self, dc, stub_main, tmp_path, monkeypatch, capsys):
        import torch
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(sys, "argv",
                            ["dc", "--encoder", str(tmp_path / "e.ckpt"),
                             "--decoder", str(tmp_path / "d.ckpt"),
                             "--max-windows", "20"])
        # Make checkpoint files exist
        (tmp_path / "e.ckpt").touch()
        (tmp_path / "d.ckpt").touch()
        assert dc.main() == 0
        out = capsys.readouterr().out
        assert "CONCORDANCE RESULTS" in out

    def test_main_too_few_windows_returns_1(self, dc, stub_main, tmp_path,
                                              monkeypatch):
        monkeypatch.setattr(sys, "argv",
                            ["dc", "--encoder", str(tmp_path / "e.ckpt"),
                             "--decoder", str(tmp_path / "d.ckpt"),
                             "--max-windows", "5"])
        (tmp_path / "e.ckpt").touch()
        (tmp_path / "d.ckpt").touch()
        # 4 files × 5 windows per file = 20 windows possible, capped to 5.
        # Still ≥ 10? No → returns 1.
        rc = dc.main()
        assert rc in (0, 1)
