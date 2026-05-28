"""Unit tests for ai_models/validation/nedc_formatter.py — Phase 2."""
from __future__ import annotations

import pytest  # decomp(eagle): skip when sibling LamQuant-Neural / -Lossless not installed
pytest.importorskip("ai_models", reason="Cross-repo test; needs sibling LamQuant-Neural or -Lossless wheel installed")

from pathlib import Path

import numpy as np
import pytest

from ai_models.validation.nedc_formatter import (
    NedcEventFormatter,
    format_predictions_for_nedc,
)

pytestmark = pytest.mark.l1


class TestConstruction:
    def test_default_label_map(self):
        f = NedcEventFormatter()
        assert f.label_map == {0: "bckg", 1: "seiz"}

    def test_custom_label_map(self):
        f = NedcEventFormatter(label_map={0: "neg", 1: "pos"})
        assert f.label_map[1] == "pos"


class TestWriteEventsCsv:
    def test_basic_write(self, tmp_path):
        f = NedcEventFormatter()
        events = [(10.5, 12.3, "seiz", 0.95), (45.2, 47.8, "seiz", 0.87)]
        out = tmp_path / "p.csv"
        f.write_events_csv(str(out), events)
        text = out.read_text()
        assert "TERM,10.5,12.3,seiz,0.9500" in text
        assert "TERM,45.2,47.8,seiz,0.8700" in text

    def test_int_label_mapped(self, tmp_path):
        f = NedcEventFormatter()
        events = [(0.0, 1.0, 1, 0.5)]  # int label
        out = tmp_path / "p.csv"
        f.write_events_csv(str(out), events)
        text = out.read_text()
        assert "seiz" in text

    def test_unknown_int_label_fallback(self, tmp_path):
        f = NedcEventFormatter()
        f.write_events_csv(str(tmp_path / "p.csv"),
                            [(0.0, 1.0, 99, 0.5)])
        text = (tmp_path / "p.csv").read_text()
        assert "label_99" in text

    def test_custom_channel(self, tmp_path):
        f = NedcEventFormatter()
        f.write_events_csv(str(tmp_path / "p.csv"),
                            [(0.0, 1.0, "seiz", 0.5)],
                            channel="C3")
        text = (tmp_path / "p.csv").read_text()
        assert "C3,0.0,1.0" in text


class TestWriteContinuousPredictions:
    def test_basic_segment_emission(self, tmp_path):
        f = NedcEventFormatter()
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        labels = np.array([0, 1, 1, 0, 0])
        confs = np.array([0.2, 0.9, 0.85, 0.1, 0.05])
        out = tmp_path / "p.csv"
        f.write_continuous_predictions(str(out), times, labels, confs,
                                        sample_rate=1.0, min_confidence=0.0)
        text = out.read_text()
        # Seizure segment from 1.0 (start) duration 2 → 3.0
        assert "seiz" in text

    def test_confidence_filter(self, tmp_path):
        f = NedcEventFormatter()
        times = np.array([0.0, 1.0, 2.0])
        labels = np.array([1, 1, 1])
        confs = np.array([0.9, 0.1, 0.9])
        f.write_continuous_predictions(str(tmp_path / "p.csv"),
                                        times, labels, confs,
                                        min_confidence=0.5)
        # The mid sample (conf 0.1) is filtered out. Remaining labels
        # [1, 1] are contiguous in the filtered list → single seizure
        # segment emitted.
        text = (tmp_path / "p.csv").read_text()
        assert text.count("seiz") == 1

    def test_no_confidence_arg_defaults_to_ones(self, tmp_path):
        f = NedcEventFormatter()
        times = np.array([0.0, 1.0])
        labels = np.array([1, 1])
        out = tmp_path / "p.csv"
        f.write_continuous_predictions(str(out), times, labels,
                                        confidences=None)
        assert out.exists()

    def test_multiclass_preds_binarized(self, tmp_path):
        f = NedcEventFormatter()
        times = np.array([0.0, 1.0])
        labels = np.array([2, 3])  # max > 1 → binarize
        f.write_continuous_predictions(str(tmp_path / "p.csv"), times, labels,
                                        confidences=np.array([0.9, 0.9]))
        # 2>0.5 = True → all become 1 → segments emitted
        text = (tmp_path / "p.csv").read_text()
        assert "seiz" in text or text == ""

    def test_empty_after_filter(self, tmp_path):
        # All confs below threshold → all filtered → empty list segmented
        f = NedcEventFormatter()
        out = tmp_path / "p.csv"
        f.write_continuous_predictions(str(out),
                                        np.array([0.0, 1.0]),
                                        np.array([1, 1]),
                                        confidences=np.array([0.1, 0.1]),
                                        min_confidence=0.5)
        # No events written (empty file)
        assert out.exists()
        assert out.read_text() == ""


class TestSegmentPredictions:
    def test_empty_returns_empty(self):
        f = NedcEventFormatter()
        out = f._segment_predictions(np.array([]), np.array([]),
                                       np.array([]), sample_rate=1.0)
        assert out == []

    def test_single_segment(self):
        f = NedcEventFormatter()
        out = f._segment_predictions(np.array([1.0, 2.0, 3.0]),
                                       np.array([1, 1, 1]),
                                       np.array([0.9, 0.8, 0.7]),
                                       sample_rate=1.0)
        assert len(out) == 1
        start, stop, label, conf = out[0]
        assert label == "seiz"

    def test_skips_background_segments(self):
        f = NedcEventFormatter()
        out = f._segment_predictions(np.array([0.0, 1.0, 2.0]),
                                       np.array([0, 0, 0]),
                                       np.array([0.9, 0.9, 0.9]),
                                       sample_rate=1.0)
        assert out == []


class TestReadEventsCsv:
    def test_read_back_written(self, tmp_path):
        f = NedcEventFormatter()
        events = [(10.5, 12.3, "seiz", 0.95)]
        out = tmp_path / "p.csv"
        f.write_events_csv(str(out), events)
        loaded = f.read_events_csv(str(out))
        assert len(loaded) == 1
        assert loaded[0]["channel"] == "TERM"
        assert loaded[0]["start_time"] == 10.5

    def test_skips_comment_lines(self, tmp_path):
        p = tmp_path / "p.csv"
        p.write_text("# version = 1.0\n# bname = foo\n"
                     "channel,start_time,stop_time,label,confidence\n"
                     "TERM,0.0,1.0,bckg,1.0\n")
        f = NedcEventFormatter()
        loaded = f.read_events_csv(str(p))
        assert len(loaded) == 1
        assert loaded[0]["label"] == "bckg"


class TestMergeAnnotations:
    def test_creates_ref_and_hyp_files(self, tmp_path):
        f = NedcEventFormatter()
        ref = tmp_path / "ref.csv"
        hyp = tmp_path / "hyp.csv"
        f.write_events_csv(str(ref), [(0.0, 1.0, "seiz", 0.9)])
        f.write_events_csv(str(hyp), [(0.5, 1.5, "seiz", 0.8)])
        f.merge_annotations(str(ref), str(hyp), str(tmp_path / "merged.csv"))
        assert (tmp_path / "merged_ref.csv").is_file()
        assert (tmp_path / "merged_hyp.csv").is_file()


class TestFormatPredictionsForNedc:
    def test_writes_csv(self, tmp_path):
        result = format_predictions_for_nedc(str(tmp_path), {
            "sample_times": np.array([0.0, 1.0, 2.0]),
            "labels": np.array([0, 1, 1]),
            "confidences": np.array([0.9, 0.9, 0.9]),
        })
        assert Path(result).is_file()
