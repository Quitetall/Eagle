#!/usr/bin/env python3
"""
LamQuant Gen 7.5 — Detail Coefficient Sparsity Diagnostic (D12)
================================================================
Question: Are detail subbands sparse enough for cheap encoding?

Tests the sparsity of L1/L2/L3 detail subbands produced by the 3-level
Le Gall 5/3 lifting DWT applied to HP-filtered EEG. Sparse detail subbands
mean fewer non-negligible coefficients, enabling cheaper Golomb-Rice or
run-length encoding in the firmware packet.

Algorithm:
  1. Load raw signal from patient NPZ (Q31 int32), convert to float
  2. Apply 2nd-order Butterworth HP filter at 0.5 Hz
  3. Extract 2500-sample windows (20 evenly spaced per file)
  4. Per window, per channel: run lifting_3level_forward
  5. Threshold each detail subband at alpha * std(detail), alpha=0.5
  6. Sparsity = 1.0 - (count_above_threshold / total_coefficients)
  7. Report mean sparsity per subband (L1, L2, L3) and per patient

PASS criterion: L1 detail mean sparsity > 70%

Usage:
  python benchmark_detail_sparsity.py
"""

import os
import sys
import numpy as np
from pathlib import Path
from scipy.signal import butter, sosfilt


def find_project_root(marker='.git'):
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / marker).exists():
            return str(parent)
    raise RuntimeError("Project root not found")


ROOT_DIR = find_project_root()
sys.path.append(os.path.join(ROOT_DIR, 'ai_models', 'student'))
from subband_preprocess import lifting_3level_forward


# Held-out patients (not used in training)
PATIENTS = {
    'chb15': 'chbmit_chb15_01_q31.npz',
    'chb16': 'chbmit_chb16_01_q31.npz',
    'chb17': 'chbmit_chb17a_03_q31.npz',
    'chb18': 'chbmit_chb18_01_q31.npz',
    'chb19': 'chbmit_chb19_01_q31.npz',
    'chb20': 'chbmit_chb20_01_q31.npz',
}

WINDOW_SIZE = 2500      # samples at 250 Hz = 10 seconds
MAX_WINDOWS = 20        # evenly spaced windows per file
ALPHA = 0.5             # sparsity threshold = alpha * std(detail)
PASS_THRESHOLD = 0.70   # L1 detail mean sparsity must exceed this

SUBBAND_NAMES = ['l1_detail', 'l2_detail', 'l3_detail']
SUBBAND_LABELS = {
    'l1_detail': 'L1 (62.5-125 Hz)',
    'l2_detail': 'L2 (31.25-62.5 Hz)',
    'l3_detail': 'L3 (15.6-31.25 Hz)',
}


def load_patient_data(path):
    """Load raw Q31 data and seizure mask from NPZ file.

    Returns:
        data: [21, T] float64, converted from Q31 int32 to microvolts
        seizure_mask: [T] float32
    """
    with np.load(path) as f:
        raw = f['data'].astype(np.float64)         # [21, T] int32 → float64
        seizure_mask = f['seizure_mask'].astype(np.float32)  # [T]

    # Q31 → physical units (microvolts scale)
    signal_float = raw / 2147483647.0 * 1000.0

    return signal_float, seizure_mask


def hp_filter(signal):
    """Apply 2nd-order Butterworth highpass at 0.5 Hz.

    Args:
        signal: [21, T] float64
    Returns:
        filtered: [21, T] float64
    """
    sos = butter(2, 0.5, btype='high', fs=250.0, output='sos')
    filtered = np.stack([sosfilt(sos, signal[ch]) for ch in range(signal.shape[0])])
    return filtered


def compute_sparsity(detail, alpha=ALPHA):
    """Compute sparsity of a detail coefficient array.

    Sparsity = fraction of coefficients at or below alpha * std(detail).

    Args:
        detail: 1D numpy array of detail coefficients
        alpha: threshold multiplier
    Returns:
        float in [0, 1]
    """
    if len(detail) == 0:
        return 0.0
    threshold = alpha * np.std(detail)
    count_above = np.sum(np.abs(detail) > threshold)
    return 1.0 - (count_above / len(detail))


def run():
    q31_dir = os.path.join(ROOT_DIR, 'ai_models', 'dataset_sim', 'q31_events')

    available = {}
    for name, fname in PATIENTS.items():
        p = os.path.join(q31_dir, fname)
        if os.path.exists(p):
            available[name] = p

    if len(available) < 3:
        print(f"[SKIP] Need at least 3 held-out patients, found {len(available)}.")
        return None

    print(f"[*] Detail Coefficient Sparsity Diagnostic (D12)")
    print(f"    Patients:      {', '.join(sorted(available.keys()))}")
    print(f"    Window size:   {WINDOW_SIZE} samples (10 s at 250 Hz)")
    print(f"    Windows/file:  {MAX_WINDOWS} evenly spaced")
    print(f"    Threshold:     alpha={ALPHA} * std(detail)")
    print(f"    PASS:          L1 sparsity > {PASS_THRESHOLD*100:.0f}%")
    print()

    # Per-patient, per-subband sparsity accumulator
    # {subband: {patient: [sparsity_values]}}
    results = {sb: {} for sb in SUBBAND_NAMES}

    for patient, path in sorted(available.items()):
        print(f"  Processing {patient} ...", end='', flush=True)

        try:
            signal_float, _ = load_patient_data(path)
        except Exception as e:
            print(f" ERROR loading: {e}")
            continue

        T = signal_float.shape[1]
        if T < WINDOW_SIZE:
            print(f" SKIP (too short: {T} samples)")
            continue

        hp_signal = hp_filter(signal_float)  # [21, T]

        # Evenly spaced window start indices
        max_start = T - WINDOW_SIZE
        if max_start == 0:
            starts = [0]
        else:
            starts = np.linspace(0, max_start, MAX_WINDOWS, dtype=int)

        patient_sparsity = {sb: [] for sb in SUBBAND_NAMES}

        for start in starts:
            window = hp_signal[:, start:start + WINDOW_SIZE]  # [21, 2500]

            for ch in range(21):
                subs = lifting_3level_forward(window[ch])
                for sb in SUBBAND_NAMES:
                    sp = compute_sparsity(subs[sb])
                    patient_sparsity[sb].append(sp)

        for sb in SUBBAND_NAMES:
            results[sb][patient] = np.array(patient_sparsity[sb])

        n_windows_done = len(starts)
        print(f" {n_windows_done} windows x 21 ch = {n_windows_done * 21} samples done.")

    # === Report ===
    print()
    print("=" * 78)
    print(" D12: DETAIL COEFFICIENT SPARSITY")
    print(f" alpha = {ALPHA}, threshold = alpha * std(detail)")
    print("=" * 78)
    print()
    print(f"{'Subband':<24} ", end='')
    patients_sorted = sorted(available.keys())
    for p in patients_sorted:
        print(f"{p:>10}", end='')
    print(f"  {'MEAN':>10}")
    print("-" * 78)

    overall_means = {}
    for sb in SUBBAND_NAMES:
        label = SUBBAND_LABELS[sb]
        print(f"{label:<24} ", end='')
        patient_means = []
        for p in patients_sorted:
            if p in results[sb] and len(results[sb][p]) > 0:
                m = np.mean(results[sb][p])
                patient_means.append(m)
                print(f"{m*100:>9.1f}%", end='')
            else:
                print(f"{'N/A':>10}", end='')
        overall = np.mean(patient_means) if patient_means else float('nan')
        overall_means[sb] = overall
        print(f"  {overall*100:>9.1f}%")

    print("-" * 78)
    print()

    # Per-patient breakdown
    print(f"{'Patient':<10} {'L1 sparsity':>14} {'L2 sparsity':>14} {'L3 sparsity':>14}")
    print("-" * 56)
    for p in patients_sorted:
        row = [p]
        for sb in SUBBAND_NAMES:
            if p in results[sb] and len(results[sb][p]) > 0:
                row.append(f"{np.mean(results[sb][p])*100:>13.1f}%")
            else:
                row.append(f"{'N/A':>14}")
        print(f"{row[0]:<10} {row[1]} {row[2]} {row[3]}")
    print()

    # Histogram summary for L1 detail (most relevant for compression)
    l1_all = []
    for p in patients_sorted:
        if p in results['l1_detail']:
            l1_all.extend(results['l1_detail'][p])

    if l1_all:
        l1_arr = np.array(l1_all)
        print(f"  L1 detail sparsity distribution:")
        print(f"    min: {l1_arr.min()*100:.1f}%   p25: {np.percentile(l1_arr,25)*100:.1f}%   "
              f"median: {np.median(l1_arr)*100:.1f}%   p75: {np.percentile(l1_arr,75)*100:.1f}%   "
              f"max: {l1_arr.max()*100:.1f}%")
        print()

    # Verdict
    l1_mean = overall_means.get('l1_detail', float('nan'))
    passed = l1_mean > PASS_THRESHOLD

    if passed:
        verdict = "PASS"
        interpretation = (
            f"L1 detail mean sparsity {l1_mean*100:.1f}% > {PASS_THRESHOLD*100:.0f}% threshold. "
            f"Golomb-Rice or run-length encoding of L1 detail will yield significant "
            f"compression gains. The wavelet transform is producing a useful sparse "
            f"representation of high-frequency EEG content."
        )
    else:
        verdict = "FAIL"
        interpretation = (
            f"L1 detail mean sparsity {l1_mean*100:.1f}% <= {PASS_THRESHOLD*100:.0f}% threshold. "
            f"High-frequency content is NOT sparse under the current lifting DWT. "
            f"Consider: (1) increasing the threshold alpha, (2) using a higher-order "
            f"filter, or (3) accepting lower compression for L1 detail."
        )

    print(f"  VERDICT: {verdict}")
    print(f"  {interpretation}")
    print()
    print("=" * 78)

    return {
        'passed': passed,
        'verdict': verdict,
        'l1_sparsity_mean': float(l1_mean),
        'l2_sparsity_mean': float(overall_means.get('l2_detail', float('nan'))),
        'l3_sparsity_mean': float(overall_means.get('l3_detail', float('nan'))),
        'per_patient': {
            sb: {p: float(np.mean(results[sb][p]))
                 for p in patients_sorted
                 if p in results[sb] and len(results[sb][p]) > 0}
            for sb in SUBBAND_NAMES
        },
    }


if __name__ == '__main__':
    result = run()
    if result is not None:
        sys.exit(0 if result['passed'] else 1)
