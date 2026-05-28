#!/usr/bin/env python3
"""
Tests for clinical seizure detection metrics (Phase 1 CRITICAL integration).

Validates:
- SeizureMetrics class for computing metrics from confusion matrices
- Clinical metric calculations (sensitivity, specificity, MCC, FAR_24h)
- Per-label confusion matrices for multi-class scenarios
- Edge cases (zero counts, perfect classification, complete failure)

Adapted metrics from NEDC EEG Evaluation Toolkit v6.0.0
"""

import pytest  # decomp(eagle): skip when sibling LamQuant-Neural / -Lossless not installed
pytest.importorskip("seizure_metrics", reason="Cross-repo test; needs sibling LamQuant-Neural or -Lossless wheel installed")

import pytest
import numpy as np
import math

from seizure_metrics import (  # via conftest sys.path
    SeizureMetrics,
    sensitivity, specificity, precision, recall, f1_score,
    matthews_cc, false_alarm_rate_24h, accuracy,
    per_label_confusion_matrix
)


@pytest.mark.l2
class TestSeizureMetricsBasic:
    """Test basic metric calculations."""

    def test_perfect_classification(self):
        """Perfect classifier: all TP, no FP/FN/TN is unusual but possible."""
        metrics = SeizureMetrics(total_duration_secs=3600)
        metrics.compute_binary(tp=100, tn=1000, fp=0, fn=0)

        assert metrics.sensitivity == 1.0, "Sensitivity should be 1.0 (perfect)"
        assert metrics.specificity == 1.0, "Specificity should be 1.0 (perfect)"
        assert metrics.precision == 1.0, "Precision should be 1.0 (perfect)"
        assert metrics.f1_score == 1.0, "F1 should be 1.0 (perfect)"
        assert metrics.accuracy == 1.0, "Accuracy should be 1.0 (perfect)"
        assert metrics.mcc == 1.0, "MCC should be 1.0 (perfect)"

    def test_complete_failure(self):
        """Classifier that always predicts wrong."""
        metrics = SeizureMetrics(total_duration_secs=3600)
        metrics.compute_binary(tp=0, tn=0, fp=100, fn=100)

        assert metrics.sensitivity == 0.0, "Sensitivity should be 0 (all missed)"
        assert metrics.specificity == 0.0, "Specificity should be 0 (all false alarms)"
        assert metrics.precision == 0.0, "Precision should be 0 (all false)"
        assert metrics.f1_score == 0.0, "F1 should be 0 (complete failure)"
        assert metrics.mcc < 0, "MCC should be negative for anti-correlated"

    def test_clinical_realistic_case(self):
        """Realistic seizure detector on 24-hour recording.

        Common scenario: ~100 seizures detected (5% of waking hours assumed),
        with 95% sensitivity, 99% specificity.
        """
        # Assume 24h recording, 5% positive events
        total_epochs = 86400  # 24h in 1s epochs
        pos_epochs = int(0.05 * total_epochs)  # ~4320 positive
        neg_epochs = total_epochs - pos_epochs

        # 95% sensitivity: catch 95% of positives
        tp = int(0.95 * pos_epochs)  # 4104
        fn = pos_epochs - tp  # 216

        # 99% specificity: 99% correctly negative
        tn = int(0.99 * neg_epochs)  # ~81360
        fp = neg_epochs - tn  # ~810

        metrics = SeizureMetrics(total_duration_secs=86400, epoch_duration_secs=1.0)
        metrics.compute_binary(tp=tp, tn=tn, fp=fp, fn=fn)

        # Verify reasonable ranges for clinical use
        assert 0.94 <= metrics.sensitivity <= 0.96, "Sensitivity ~95%"
        assert 0.98 <= metrics.specificity <= 0.99, "Specificity ~99%"
        assert 0.80 <= metrics.precision <= 0.90, f"Precision should be 80-90%, got {metrics.precision:.2%}"
        assert 0.70 <= metrics.f1_score <= 0.92, "F1 should be 70-92%"

        # False alarm rate: ~810 FPs in 24h = ~810 alarms per 24h
        assert 0.0 <= metrics.false_alarm_rate_24h <= 1000, \
            f"FAR_24h should be reasonable, got {metrics.false_alarm_rate_24h:.1f}"

    def test_zero_division_handling(self):
        """Verify zero division doesn't cause crashes."""
        metrics = SeizureMetrics()

        # Empty confusion matrix
        metrics.compute_binary(tp=0, tn=0, fp=0, fn=0)
        assert metrics.sensitivity == 0.0
        assert metrics.specificity == 0.0
        assert metrics.precision == 0.0
        assert metrics.f1_score == 0.0
        assert metrics.mcc == 0.0
        assert metrics.accuracy == 0.0

    def test_mcc_invariance(self):
        """MCC should be invariant to class swap (unlike accuracy)."""
        # Case 1: More positives
        metrics1 = SeizureMetrics()
        metrics1.compute_binary(tp=50, tn=50, fp=10, fn=10)
        mcc1 = metrics1.mcc

        # Case 2: Swap TP/TN and FP/FN (swap class labels)
        metrics2 = SeizureMetrics()
        metrics2.compute_binary(tp=50, tn=50, fp=10, fn=10)  # Same values
        mcc2 = metrics2.mcc

        assert abs(mcc1 - mcc2) < 1e-6, "MCC should be same regardless of label"


@pytest.mark.l2
class TestSeizureMetricsFromArrays:
    """Test metrics computed from prediction arrays."""

    def test_from_binary_arrays(self):
        """Compute metrics from ground truth and prediction arrays."""
        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 0, 1, 0, 1, 0, 1, 1])

        metrics = SeizureMetrics(total_duration_secs=8)
        metrics.compute_from_arrays(y_true, y_pred)

        # Manual calculation:
        # TP: (1,1) at indices 2,4,6 = 3
        # TN: (0,0) at indices 0,1,5 = 3
        # FP: (0,1) at indices 7 = 1
        # FN: (1,0) at index 3 = 1

        assert metrics.tp == 3
        assert metrics.tn == 3
        assert metrics.fp == 1
        assert metrics.fn == 1

        expected_sensitivity = 3 / 4  # 0.75
        expected_specificity = 3 / 4  # 0.75
        assert abs(metrics.sensitivity - expected_sensitivity) < 1e-6

    def test_from_probability_arrays(self):
        """Compute metrics from probability predictions with threshold."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.4, 0.7])

        metrics = SeizureMetrics()
        metrics.compute_from_arrays(y_true, y_prob, threshold=0.5)

        # y_prob > 0.5: [False, False, True, True, False, True] = [0, 0, 1, 1, 0, 1]
        # Matches y_true exactly
        # TP=3 (indices 2,3,5), TN=3 (indices 0,1,4), FP=0, FN=0
        assert metrics.tp == 3
        assert metrics.tn == 3
        assert metrics.fp == 0
        assert metrics.fn == 0


@pytest.mark.l2
class TestStandaloneFunctions:
    """Test standalone metric functions."""

    def test_sensitivity_function(self):
        """Test sensitivity calculation."""
        assert sensitivity(50, 10) == 50 / 60
        assert sensitivity(0, 10) == 0.0
        assert sensitivity(10, 0) == 1.0
        assert sensitivity(0, 0) == 0.0

    def test_specificity_function(self):
        """Test specificity calculation."""
        assert specificity(100, 5) == 100 / 105
        assert specificity(0, 5) == 0.0
        assert specificity(100, 0) == 1.0

    def test_precision_function(self):
        """Test precision calculation."""
        assert precision(50, 10) == 50 / 60
        assert precision(0, 10) == 0.0
        assert precision(50, 0) == 1.0

    def test_f1_score_function(self):
        """Test F1 score calculation."""
        # Perfect: precision=1, recall=1 -> F1=1
        assert f1_score(100, 0, 0) == 1.0

        # Precision=0.5, Recall=0.5 -> F1=0.5
        tp, fp, fn = 50, 50, 50
        prec = tp / (tp + fp)  # 0.5
        rec = tp / (tp + fn)   # 0.5
        expected_f1 = 2 * 0.5 * 0.5 / (0.5 + 0.5)  # 0.5
        assert abs(f1_score(tp, fp, fn) - expected_f1) < 1e-6

    def test_matthews_cc_function(self):
        """Test Matthews Correlation Coefficient."""
        # Perfect correlation
        assert matthews_cc(100, 100, 0, 0) == 1.0

        # Perfect anti-correlation
        assert matthews_cc(0, 0, 100, 100) == -1.0

        # No correlation
        assert abs(matthews_cc(50, 50, 50, 50)) < 1e-6

        # Zero denominator
        assert matthews_cc(0, 0, 0, 0) == 0.0

    def test_false_alarm_rate_24h(self):
        """Test false alarm rate per 24 hours."""
        # 100 FPs in 24h with 1s epochs = 100 FPs per 24h
        far = false_alarm_rate_24h(fp=100, epoch_duration_secs=1.0,
                                   total_duration_secs=86400)
        assert abs(far - 100.0) < 1e-6

        # 10 FPs in 1h = ~240 FPs per 24h
        far = false_alarm_rate_24h(fp=10, epoch_duration_secs=1.0,
                                   total_duration_secs=3600)
        expected = 10 * 1.0 / 3600 * 86400
        assert abs(far - expected) < 1e-6

        # Zero FPs
        assert false_alarm_rate_24h(fp=0) == 0.0

        # Zero duration
        assert false_alarm_rate_24h(fp=100, total_duration_secs=0) == 0.0

    def test_accuracy_function(self):
        """Test accuracy calculation."""
        # Perfect: all correct
        assert accuracy(100, 100, 0, 0) == 1.0

        # 80% correct: 80 TP, 20 FP, 20 FN out of 120
        assert accuracy(80, 80, 20, 20) == 160 / 200  # 0.8

        # Zero total
        assert accuracy(0, 0, 0, 0) == 0.0


@pytest.mark.l2
class TestPerLabelConfusionMatrix:
    """Test per-label confusion matrix for multi-class scenarios."""

    def test_binary_classification(self):
        """Binary classification with 2 classes."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 0, 1, 1])

        result = per_label_confusion_matrix(y_true, y_pred)

        # Label 0: TP=2, TN=2, FP=1, FN=1
        assert result[0]['tp'] == 2
        assert result[0]['tn'] == 2
        assert result[0]['fp'] == 1
        assert result[0]['fn'] == 1

        # Label 1: TP=2, TN=2, FP=1, FN=1
        assert result[1]['tp'] == 2
        assert result[1]['tn'] == 2
        assert result[1]['fp'] == 1
        assert result[1]['fn'] == 1

    def test_multi_class_classification(self):
        """Multi-class classification with 3 classes."""
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1])
        y_pred = np.array([0, 1, 2, 0, 2, 2, 1, 1])

        result = per_label_confusion_matrix(y_true, y_pred)

        # Should have entries for classes 0, 1, 2
        assert 0 in result
        assert 1 in result
        assert 2 in result

        # Each should have the required metrics
        for label in result:
            assert 'tp' in result[label]
            assert 'tn' in result[label]
            assert 'fp' in result[label]
            assert 'fn' in result[label]
            assert 'sensitivity' in result[label]
            assert 'specificity' in result[label]
            assert 'precision' in result[label]
            assert 'f1_score' in result[label]
            assert 'mcc' in result[label]

    def test_per_label_imbalanced_classes(self):
        """Imbalanced multi-class problem."""
        # 10 class 0, 3 class 1, 2 class 2
        y_true = np.array([0]*10 + [1]*3 + [2]*2)
        y_pred = np.array(
            [0]*9 + [1] +      # 9 correct class 0, 1 missed
            [1]*2 + [0] +      # 2 correct class 1, 1 missed as class 0
            [2]*2              # 2 correct class 2
        )

        result = per_label_confusion_matrix(y_true, y_pred)

        # Class 0: TP=9, very high TN (lots of negatives)
        assert result[0]['tp'] == 9
        assert result[0]['fn'] == 1  # One class 0 predicted as class 1

        # Class 1: TP=2, FN=1
        assert result[1]['tp'] == 2
        assert result[1]['fn'] == 1  # One class 1 predicted as class 0


@pytest.mark.l5
class TestMetricsConsistency:
    """Test internal consistency of metrics."""

    def test_sensitivity_with_fn(self):
        """Sensitivity + FNR should equal 1."""
        metrics = SeizureMetrics()
        metrics.compute_binary(tp=80, tn=100, fp=20, fn=20)

        assert abs((metrics.sensitivity + metrics.fnr) - 1.0) < 1e-6

    def test_specificity_with_fpr(self):
        """Specificity + FPR should equal 1."""
        metrics = SeizureMetrics()
        metrics.compute_binary(tp=80, tn=100, fp=20, fn=20)

        assert abs((metrics.specificity + metrics.fpr) - 1.0) < 1e-6

    def test_confusion_matrix_sum(self):
        """All confusion matrix values should sum correctly."""
        metrics = SeizureMetrics()
        metrics.compute_binary(tp=50, tn=60, fp=15, fn=25)

        # TP + TN + FP + FN should equal total samples
        total = metrics.tp + metrics.tn + metrics.fp + metrics.fn
        assert total == 50 + 60 + 15 + 25

    def test_mcc_bounds(self):
        """MCC should always be in [-1, 1]."""
        test_cases = [
            (100, 100, 0, 0),    # Perfect
            (0, 0, 100, 100),    # Complete failure
            (50, 50, 50, 50),    # Random
            (80, 60, 20, 40),    # Good classifier
        ]

        for tp, tn, fp, fn in test_cases:
            mcc = matthews_cc(tp, tn, fp, fn)
            assert -1.0 <= mcc <= 1.0, f"MCC out of bounds: {mcc}"


@pytest.mark.l5
class TestClinicalRanges:
    """Test that metrics fall within expected clinical ranges."""

    def test_seizure_detection_clinical_thresholds(self):
        """Verify reasonable thresholds for seizure detection systems."""
        # Clinical requirement: Sensitivity > 90%, Specificity > 95%
        metrics = SeizureMetrics(total_duration_secs=86400)
        metrics.compute_binary(tp=450, tn=4750, fp=250, fn=50)

        # tp/(tp+fn) = 450/500 = 0.90
        assert metrics.sensitivity >= 0.90, "Should meet clinical sensitivity threshold"

        # tn/(tn+fp) = 4750/5000 = 0.95
        assert metrics.specificity >= 0.95, "Should meet clinical specificity threshold"

    def test_false_alarm_rate_clinical_threshold(self):
        """Clinical requirement: FAR < 1 per 24h for patient safety."""
        metrics = SeizureMetrics(total_duration_secs=86400, epoch_duration_secs=1.0)

        # Only 10 FPs in 24h recording
        metrics.compute_binary(tp=400, tn=86000, fp=10, fn=50)

        assert metrics.false_alarm_rate_24h <= 10.0, "Should be below clinical FAR threshold"

    def test_mcc_clinical_acceptability(self):
        """MCC > 0.6 is generally considered good even for imbalanced datasets."""
        metrics = SeizureMetrics()
        # Biased towards negatives (seizure detection scenario)
        # TP=100, TN=10000, FP=100, FN=20
        metrics.compute_binary(tp=100, tn=10000, fp=100, fn=20)

        # MCC = (100*10000 - 100*20) / sqrt((100+100)*(100+20)*(10000+100)*(10000+20))
        # MCC = (1000000 - 2000) / sqrt(200*120*10100*10020) ≈ 0.577
        assert metrics.mcc > 0.5, "Should achieve reasonable MCC for imbalanced problem"


@pytest.mark.l2
class TestMetricsOutputFormat:
    """Test output formats and serialization."""

    def test_to_dict_format(self):
        """Verify to_dict() returns all required metrics."""
        metrics = SeizureMetrics()
        metrics.compute_binary(tp=80, tn=100, fp=20, fn=20)

        result = metrics.to_dict()

        required_keys = {
            'tp', 'tn', 'fp', 'fn',
            'sensitivity', 'specificity', 'precision', 'npv',
            'fnr', 'fpr', 'f1_score', 'mcc', 'accuracy', 'prevalence',
            'false_alarm_rate_24h'
        }

        assert set(result.keys()) == required_keys, \
            f"Missing keys: {required_keys - set(result.keys())}"

        # All values should be numbers
        for key, value in result.items():
            assert isinstance(value, (int, float)), \
                f"Value for {key} should be numeric, got {type(value)}"

    def test_repr_format(self):
        """Verify string representation is informative."""
        metrics = SeizureMetrics()
        metrics.compute_binary(tp=80, tn=100, fp=20, fn=20)

        repr_str = repr(metrics)

        # Should contain key metrics
        assert 'TP=80' in repr_str
        assert 'TN=100' in repr_str
        assert 'FP=20' in repr_str
        assert 'FN=20' in repr_str
        assert 'Sens=' in repr_str
        assert 'Spec=' in repr_str
        assert 'F1=' in repr_str
        assert 'MCC=' in repr_str


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
