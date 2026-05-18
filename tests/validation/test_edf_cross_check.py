"""Unit tests for ai_models/validation/edf_cross_check.py — Phase 2."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from ai_models.validation.edf_cross_check import (
    ChannelDiff,
    CrossCheckResult,
    cross_check_edf,
    _read_with_ours,
    _read_with_pyedflib,
)

pytestmark = pytest.mark.l2


# ---------------------------------------------------------------------------
# ChannelDiff + CrossCheckResult
# ---------------------------------------------------------------------------
class TestChannelDiff:
    def test_basic_construction(self):
        d = ChannelDiff(label="C3", n_samples=2500,
                         max_abs_diff=1e-12, rmse=5e-13, rel_rmse=0.0)
        assert d.label == "C3"
        assert d.n_samples == 2500


class TestCrossCheckResult:
    def _result(self, **kwargs):
        defaults = dict(
            edf_path="/x.edf", sfreq_ours=250.0, sfreq_pyedflib=250.0,
            channels_ours=["C3"], channels_pyedflib=["C3"],
            channels_compared=["C3"],
            channels_only_ours=[], channels_only_pyedflib=[],
            per_channel=[],
        )
        defaults.update(kwargs)
        return CrossCheckResult(**defaults)

    def test_max_abs_diff_default_zero(self):
        r = self._result()
        assert r.max_abs_diff == 0.0

    def test_max_abs_diff_takes_max(self):
        r = self._result(per_channel=[
            ChannelDiff(label="a", n_samples=10, max_abs_diff=1e-9,
                         rmse=0.0, rel_rmse=0.0),
            ChannelDiff(label="b", n_samples=10, max_abs_diff=3e-9,
                         rmse=0.0, rel_rmse=0.0),
        ])
        assert r.max_abs_diff == 3e-9

    def test_worst_rmse(self):
        r = self._result(per_channel=[
            ChannelDiff(label="a", n_samples=10, max_abs_diff=0,
                         rmse=0.5, rel_rmse=0),
            ChannelDiff(label="b", n_samples=10, max_abs_diff=0,
                         rmse=0.1, rel_rmse=0),
        ])
        assert r.worst_rmse == 0.5

    def test_sample_rates_agree_true(self):
        assert self._result(sfreq_ours=250.0, sfreq_pyedflib=250.0).sample_rates_agree

    def test_sample_rates_disagree(self):
        assert not self._result(sfreq_ours=250.0,
                                 sfreq_pyedflib=256.0).sample_rates_agree

    def test_sample_rates_none(self):
        assert not self._result(sfreq_ours=None).sample_rates_agree

    def test_is_bit_equivalent_pass(self):
        r = self._result(per_channel=[
            ChannelDiff(label="a", n_samples=10, max_abs_diff=1e-12,
                         rmse=0.0, rel_rmse=0.0)
        ])
        assert r.is_bit_equivalent(tol=1e-9)

    def test_is_bit_equivalent_fails_above_tol(self):
        r = self._result(per_channel=[
            ChannelDiff(label="a", n_samples=10, max_abs_diff=1e-5,
                         rmse=0.0, rel_rmse=0.0)
        ])
        assert not r.is_bit_equivalent(tol=1e-9)

    def test_is_bit_equivalent_empty_per_channel_fails(self):
        # No channels compared → not bit-equivalent
        assert not self._result().is_bit_equivalent()

    def test_summary_contains_expected_keys(self):
        r = self._result(per_channel=[
            ChannelDiff(label="C3", n_samples=100, max_abs_diff=1e-10,
                         rmse=5e-11, rel_rmse=0.0)
        ])
        s = r.summary()
        assert "EDF:" in s
        assert "sfreq ours" in s
        assert "channels ours" in s
        assert "max |diff|" in s


# ---------------------------------------------------------------------------
# cross_check_edf (with mocked readers)
# ---------------------------------------------------------------------------
class TestCrossCheckEdf:
    def test_bit_equivalent_paths(self):
        sig = np.random.randn(2500)
        # Identical readers → bit-equivalent result
        with patch("ai_models.validation.edf_cross_check._read_with_ours",
                   return_value=({"C3": sig, "C4": sig}, 250.0)), \
             patch("ai_models.validation.edf_cross_check._read_with_pyedflib",
                   return_value=({"C3": sig, "C4": sig}, 250.0)):
            result = cross_check_edf("/fake.edf")
        assert result.is_bit_equivalent()
        assert len(result.per_channel) == 2
        assert result.channels_only_ours == []
        assert result.channels_only_pyedflib == []

    def test_disagreement_reported(self):
        with patch("ai_models.validation.edf_cross_check._read_with_ours",
                   return_value=({"C3": np.ones(100)}, 250.0)), \
             patch("ai_models.validation.edf_cross_check._read_with_pyedflib",
                   return_value=({"C3": np.zeros(100)}, 250.0)):
            result = cross_check_edf("/fake.edf")
        assert not result.is_bit_equivalent()
        assert result.max_abs_diff == pytest.approx(1.0)

    def test_channel_set_difference(self):
        sig = np.zeros(100)
        with patch("ai_models.validation.edf_cross_check._read_with_ours",
                   return_value=({"C3": sig, "extra": sig}, 250.0)), \
             patch("ai_models.validation.edf_cross_check._read_with_pyedflib",
                   return_value=({"C3": sig, "other": sig}, 250.0)):
            result = cross_check_edf("/fake.edf")
        assert result.channels_only_ours == ["extra"]
        assert result.channels_only_pyedflib == ["other"]

    def test_none_ours_treated_as_empty(self):
        with patch("ai_models.validation.edf_cross_check._read_with_ours",
                   return_value=(None, None)), \
             patch("ai_models.validation.edf_cross_check._read_with_pyedflib",
                   return_value=({"C3": np.zeros(10)}, 250.0)):
            result = cross_check_edf("/fake.edf")
        assert result.channels_compared == []

    def test_length_mismatch_uses_min(self):
        with patch("ai_models.validation.edf_cross_check._read_with_ours",
                   return_value=({"C3": np.ones(100)}, 250.0)), \
             patch("ai_models.validation.edf_cross_check._read_with_pyedflib",
                   return_value=({"C3": np.ones(50)}, 250.0)):
            result = cross_check_edf("/fake.edf")
        # Truncated to n=50
        assert result.per_channel[0].n_samples == 50

    def test_zero_length_channel_skipped(self):
        with patch("ai_models.validation.edf_cross_check._read_with_ours",
                   return_value=({"C3": np.array([])}, 250.0)), \
             patch("ai_models.validation.edf_cross_check._read_with_pyedflib",
                   return_value=({"C3": np.array([])}, 250.0)):
            result = cross_check_edf("/fake.edf")
        assert result.per_channel == []

    def test_flat_reference_rel_rmse_inf_when_diff_nonzero(self):
        # Reference is all zeros (flat) but ours differs → rel_rmse = inf
        with patch("ai_models.validation.edf_cross_check._read_with_ours",
                   return_value=({"C3": np.ones(50)}, 250.0)), \
             patch("ai_models.validation.edf_cross_check._read_with_pyedflib",
                   return_value=({"C3": np.zeros(50)}, 250.0)):
            result = cross_check_edf("/fake.edf")
        assert result.per_channel[0].rel_rmse == float("inf")
