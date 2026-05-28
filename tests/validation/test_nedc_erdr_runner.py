"""
Tests for the ERDR (NEDC ResNet Real-Time Decoder) subprocess wrapper.

We don't actually invoke ERDR in this test file — doing so requires torch in
the subprocess's environment and takes tens of seconds. Instead we test:

  1. `check_installation()` correctly detects an intact ERDR tree.
  2. `check_installation()` correctly flags a missing/broken tree.
  3. `parse_csvbi()` handles real NEDC csv_bi files (comments, header, rows).
  4. `events_to_labels()` produces correct per-epoch labels.
  5. The constructed driver command references files that actually exist.

Running ERDR end-to-end is a manual integration test — see the class-level
`run_once` helper at the bottom for a smoke test you can invoke by hand.
"""

import pytest  # decomp(eagle): skip when sibling LamQuant-Neural / -Lossless not installed
pytest.importorskip("ai_models", reason="Cross-repo test; needs sibling LamQuant-Neural or -Lossless wheel installed")

import os
import sys
from pathlib import Path

import pytest
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))  # moved into tests/validation/

from ai_models.validation.nedc_erdr_runner import (  # via conftest sys.path
    NedcErdrRunner,
    parse_csvbi,
)

ERDR_ROOT = os.path.join(
    _REPO_ROOT, "Reference Software", "nedc_eeg_resnet_decode_realtime", "v1.0.0",
)
REAL_CSVBI = os.path.join(
    _REPO_ROOT, "Reference Software", "nedc_eeg_eval", "v6.0.0",
    "data", "csv", "hyp", "aaaaaasf_s001_t000.csv_bi",
)


@pytest.mark.l5
class TestCheckInstallation:

    @pytest.mark.skipif(not os.path.isdir(ERDR_ROOT),
                        reason="ERDR v1.0.0 not present")
    def test_detects_intact_install(self):
        runner = NedcErdrRunner(erdr_root=ERDR_ROOT)
        status = runner.check_installation()
        assert status.ok, status.summary()
        assert status.missing == []

    def test_detects_missing_tree(self, tmp_path):
        runner = NedcErdrRunner(erdr_root=tmp_path / "does_not_exist")
        status = runner.check_installation()
        assert not status.ok
        assert len(status.missing) >= 1

    def test_detects_partial_install(self, tmp_path):
        # Empty dir = every required file missing
        (tmp_path / "erdr_empty").mkdir()
        runner = NedcErdrRunner(erdr_root=tmp_path / "erdr_empty")
        status = runner.check_installation()
        assert not status.ok
        # All REQUIRED_FILES should be reported missing
        assert len(status.missing) == len(NedcErdrRunner.REQUIRED_FILES)

    def test_notes_torch_availability(self, tmp_path):
        # This test runs from LamQuant's venv which doesn't have torch.
        # The check should still pass structurally and note the gap.
        runner = NedcErdrRunner(erdr_root=tmp_path / "whatever")
        status = runner.check_installation()
        # We just verify the `notes` field is populated or empty — both OK,
        # depending on whether torch happens to be installed in this env.
        assert isinstance(status.notes, list)


@pytest.mark.l5
class TestParseCsvbi:

    @pytest.mark.skipif(not os.path.exists(REAL_CSVBI),
                        reason="nedc_eeg_eval sample csv_bi not present")
    def test_parses_real_nedc_file(self):
        events = parse_csvbi(REAL_CSVBI)
        assert len(events) > 0
        for ev in events:
            assert set(ev.keys()) == {
                "channel", "start_time", "stop_time", "label", "confidence",
            }
            assert isinstance(ev["start_time"], float)
            assert isinstance(ev["stop_time"], float)
            assert ev["stop_time"] >= ev["start_time"]
            assert ev["label"] in ("seiz", "bckg")

    def test_parses_synthetic_file(self, tmp_path):
        path = tmp_path / "tiny.csv_bi"
        path.write_text(
            "# version = csv_v1.0.0\n"
            "# bname = test\n"
            "# duration = 30.0000 secs\n"
            "#\n"
            "channel,start_time,stop_time,label,confidence\n"
            "TERM,0.0000,10.0000,bckg,1.0000\n"
            "TERM,10.0000,20.0000,seiz,0.8500\n"
            "TERM,20.0000,30.0000,bckg,1.0000\n"
        )
        events = parse_csvbi(str(path))
        assert len(events) == 3
        assert events[1]["label"] == "seiz"
        assert events[1]["start_time"] == 10.0
        assert events[1]["stop_time"] == 20.0
        assert abs(events[1]["confidence"] - 0.85) < 1e-9

    def test_skips_malformed_rows(self, tmp_path):
        path = tmp_path / "bad.csv_bi"
        path.write_text(
            "# header\n"
            "channel,start_time,stop_time,label,confidence\n"
            "TERM,0.0,10.0,bckg,1.0\n"
            "garbage row\n"
            "TERM,oops,nope,seiz,1.0\n"
            "TERM,10.0,20.0,seiz,0.9\n"
        )
        events = parse_csvbi(str(path))
        assert len(events) == 2  # only the two well-formed rows


@pytest.mark.l2
class TestEventsToLabels:

    def _events(self):
        return [
            {"channel": "TERM", "start_time": 0.0, "stop_time": 10.0,
             "label": "bckg", "confidence": 1.0},
            {"channel": "TERM", "start_time": 10.0, "stop_time": 20.0,
             "label": "seiz", "confidence": 0.9},
            {"channel": "TERM", "start_time": 20.0, "stop_time": 30.0,
             "label": "bckg", "confidence": 1.0},
        ]

    def test_basic_1s_epochs(self):
        y, dur = NedcErdrRunner.events_to_labels(self._events(), epoch=1.0)
        assert dur == 30.0
        assert y.shape == (30,)
        assert y.sum() == 10  # 10 s of seizure
        assert (y[0:10] == 0).all()
        assert (y[10:20] == 1).all()
        assert (y[20:30] == 0).all()

    def test_explicit_duration(self):
        y, dur = NedcErdrRunner.events_to_labels(
            self._events(), duration_s=60.0, epoch=1.0,
        )
        assert dur == 60.0
        assert y.shape == (60,)
        assert y.sum() == 10  # same seizure, longer background tail

    def test_half_second_epochs(self):
        y, _ = NedcErdrRunner.events_to_labels(self._events(), epoch=0.5)
        assert y.shape == (60,)
        assert y.sum() == 20  # 10 s of seizure @ 0.5 s/bin

    def test_empty_events(self):
        y, dur = NedcErdrRunner.events_to_labels([])
        assert dur == 0.0
        assert y.shape == (0,)


@pytest.mark.l5
class TestDriverCommand:
    """Sanity: make sure check_installation + _build_driver_command reference
    files that actually exist on disk."""

    @pytest.mark.skipif(not os.path.isdir(ERDR_ROOT),
                        reason="ERDR v1.0.0 not present")
    def test_driver_command_points_to_real_files(self):
        runner = NedcErdrRunner(erdr_root=ERDR_ROOT)
        cmd = runner._build_driver_command(
            edf_path="/nonexistent.edf",
            output_dir="/tmp",
            basename="x",
        )
        # cmd = [python, driver.py, -p, params, -o, /tmp, -r, None, -b, x, edf]
        driver_py = cmd[1]
        params_txt = cmd[3]
        assert os.path.exists(driver_py), f"missing driver: {driver_py}"
        assert os.path.exists(params_txt), f"missing params: {params_txt}"


# ============================================================
# Coverage supplement: summary, env_root, decode()
# ============================================================

class TestInstallationStatusSummary:
    def test_summary_includes_status_and_root(self):
        from ai_models.validation.nedc_erdr_runner import InstallationStatus
        s = InstallationStatus(
            root=Path("/x"), ok=True, missing=[], notes=[])
        out = s.summary()
        assert "/x" in out
        assert "OK" in out

    def test_summary_lists_missing(self):
        from ai_models.validation.nedc_erdr_runner import InstallationStatus
        s = InstallationStatus(
            root=Path("/x"), ok=False,
            missing=["bin/foo", "lib/bar"],
            notes=["something"])
        out = s.summary()
        assert "INCOMPLETE" in out
        assert "bin/foo" in out
        assert "something" in out


class TestEnvRootResolution:
    def test_NEDC_NFC_env_used(self, monkeypatch, tmp_path):
        from ai_models.validation.nedc_erdr_runner import NedcErdrRunner
        monkeypatch.setenv("NEDC_NFC", str(tmp_path))
        r = NedcErdrRunner()
        assert r.root == tmp_path.resolve()

    def test_explicit_root_overrides_env(self, monkeypatch, tmp_path):
        from ai_models.validation.nedc_erdr_runner import NedcErdrRunner
        monkeypatch.setenv("NEDC_NFC", "/some/env/path")
        r = NedcErdrRunner(erdr_root=tmp_path)
        assert r.root == tmp_path.resolve()


class TestEmptyModelFile:
    def test_empty_model_pckl_added_to_missing(self, tmp_path):
        from ai_models.validation.nedc_erdr_runner import NedcErdrRunner
        # Create all required files, but make model.pckl empty
        for rel in NedcErdrRunner.REQUIRED_FILES:
            p = tmp_path / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            if "model.pckl" in rel:
                p.touch()  # 0 bytes
            else:
                p.write_text("x")
        r = NedcErdrRunner(erdr_root=tmp_path)
        status = r.check_installation()
        assert any("present but empty" in m for m in status.missing)


class TestDecode:
    def test_missing_edf_raises(self, tmp_path):
        from ai_models.validation.nedc_erdr_runner import NedcErdrRunner
        r = NedcErdrRunner(erdr_root=tmp_path)
        with pytest.raises(FileNotFoundError):
            r.decode("/nonexistent.edf")

    def test_decode_with_mocked_subprocess(self, tmp_path, monkeypatch):
        from unittest.mock import MagicMock, patch
        import ai_models.validation.nedc_erdr_runner as mod
        # Create dummy EDF
        edf = tmp_path / "x.edf"
        edf.write_text("not a real EDF")
        # Mock subprocess.run
        mock_proc = MagicMock(returncode=0, stdout="ok", stderr="")
        with patch.object(mod.subprocess, "run", return_value=mock_proc):
            r = mod.NedcErdrRunner(erdr_root=tmp_path)
            out = r.decode(str(edf), output_dir=str(tmp_path / "out"),
                            basename="test")
        assert out.returncode == 0
        assert out.edf_path == str(edf.resolve())

    def test_decode_returns_events_when_csvbi_written(self, tmp_path, monkeypatch):
        from unittest.mock import MagicMock, patch
        import ai_models.validation.nedc_erdr_runner as mod

        edf = tmp_path / "x.edf"
        edf.write_text("not a real EDF")
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        # Pre-create csv_bi file as if subprocess wrote it
        csvbi = out_dir / "test.csv_bi"
        csvbi.write_text(
            "# version = 1.0\n"
            "channel,start_time,stop_time,label,confidence\n"
            "TERM,0.0,10.0,bckg,1.0\n"
            "TERM,10.0,20.0,seiz,0.95\n"
        )

        mock_proc = MagicMock(returncode=0, stdout="", stderr="")
        with patch.object(mod.subprocess, "run", return_value=mock_proc):
            r = mod.NedcErdrRunner(erdr_root=tmp_path)
            out = r.decode(str(edf), output_dir=str(out_dir),
                            basename="test")
        assert len(out.events) == 2
        assert out.csvbi_path is not None
