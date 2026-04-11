#!/usr/bin/env python3
"""
LamQuant Gen 7.5 — LPC Prediction Gain Diagnostic (D13)
========================================================
Question: Is LPC worth its 3.1 ms latency cost?

Measures the prediction gain achieved by 8th-order LPC on HP-filtered EEG,
separately for quiescent (inter-ictal) and seizure windows. Prediction gain
is the ratio of input signal variance to residual variance, in dB.

If quiescent gain > 3 dB, LPC is compressing the signal: the residual has
lower entropy than the raw signal, making subsequent Golomb-Rice coding
cheaper. If gain is near 0 dB, LPC is not helping and its 3.1 ms latency
is pure overhead.

Algorithm:
  1. Load raw signal from patient NPZ (Q31 int32), convert to float
  2. Apply 2nd-order Butterworth HP filter at 0.5 Hz
  3. Extract 2500-sample windows (20 evenly spaced per file)
  4. Per window, per channel: lpc_analyze_channel(signal, order=8, autocorr_len=256)
  5. Prediction gain = 10 * log10(var(input) / (var(residual) + 1e-30))
  6. Classify window as seizure if any(seizure_mask > 0) for the sample range
  7. Report mean gain overall / quiescent / seizure, per patient and per channel

PASS criterion: quiescent mean prediction gain > 3 dB

Usage:
  python benchmark_lpc_gain.py
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
from subband_preprocess import lpc_analyze_channel


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
LPC_ORDER = 8           # matches firmware lpc_predictor.c
AUTOCORR_LEN = 256      # matches firmware
PASS_THRESHOLD_DB = 3.0  # quiescent mean gain must exceed this


def load_patient_data(path):
    """Load raw Q31 data and seizure mask from NPZ file.

    Returns:
        data: [21, T] float64, converted from Q31 int32
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


def prediction_gain_db(signal_ch, residual):
    """Compute LPC prediction gain in dB.

    Returns None for flat/dead channels (input variance near zero), since
    the gain formula is undefined for a zero-variance signal.

    Args:
        signal_ch: [T] input signal
        residual: [T] LPC prediction residual
    Returns:
        float: gain in dB, or None if channel is flat/dead
    """
    sig_var = np.var(signal_ch)
    if sig_var < 1e-20:
        return None  # dead/flat channel — skip
    res_var = np.var(residual)
    return 10.0 * np.log10(sig_var / (res_var + 1e-30))


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

    print(f"[*] LPC Prediction Gain Diagnostic (D13)")
    print(f"    Patients:      {', '.join(sorted(available.keys()))}")
    print(f"    Window size:   {WINDOW_SIZE} samples (10 s at 250 Hz)")
    print(f"    Windows/file:  {MAX_WINDOWS} evenly spaced")
    print(f"    LPC order:     {LPC_ORDER}  (autocorr_len={AUTOCORR_LEN})")
    print(f"    PASS:          quiescent gain > {PASS_THRESHOLD_DB:.1f} dB")
    print()

    # Accumulators: per patient → lists of (gain, is_seizure) tuples
    patient_results = {}

    for patient, path in sorted(available.items()):
        print(f"  Processing {patient} ...", end='', flush=True)

        try:
            signal_float, seizure_mask = load_patient_data(path)
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

        gains_all = []        # (gain_db, is_seizure, channel)
        seizure_count = 0
        quiescent_count = 0

        for start in starts:
            end = start + WINDOW_SIZE
            mask_window = seizure_mask[start:end]
            is_seizure = bool(np.any(mask_window > 0))

            if is_seizure:
                seizure_count += 1
            else:
                quiescent_count += 1

            window = hp_signal[:, start:end]  # [21, 2500]

            for ch in range(21):
                coeffs, residual = lpc_analyze_channel(
                    window[ch], order=LPC_ORDER, autocorr_len=AUTOCORR_LEN
                )
                gain = prediction_gain_db(window[ch], residual)
                if gain is None:
                    continue  # skip dead/flat channels
                gains_all.append({
                    'gain': gain,
                    'is_seizure': is_seizure,
                    'channel': ch,
                })

        patient_results[patient] = gains_all
        n_windows = len(starts)
        print(f" {n_windows} windows ({seizure_count} seizure, {quiescent_count} quiescent) x 21 ch done.")

    # === Aggregate results ===
    patients_sorted = sorted(available.keys())

    # Per-patient summary
    summary = {}
    for patient in patients_sorted:
        if patient not in patient_results:
            continue
        gains = patient_results[patient]
        all_gains = [g['gain'] for g in gains]
        qgains = [g['gain'] for g in gains if not g['is_seizure']]
        sgains = [g['gain'] for g in gains if g['is_seizure']]

        summary[patient] = {
            'mean_all': np.mean(all_gains) if all_gains else float('nan'),
            'mean_quiescent': np.mean(qgains) if qgains else float('nan'),
            'mean_seizure': np.mean(sgains) if sgains else float('nan'),
            'n_quiescent': len(qgains) // 21,   # window count
            'n_seizure': len(sgains) // 21,
        }

    # Per-channel summary (across all patients)
    channel_gains = {ch: [] for ch in range(21)}
    channel_gains_q = {ch: [] for ch in range(21)}
    channel_gains_s = {ch: [] for ch in range(21)}
    for patient in patients_sorted:
        if patient not in patient_results:
            continue
        for g in patient_results[patient]:
            ch = g['channel']
            channel_gains[ch].append(g['gain'])
            if g['is_seizure']:
                channel_gains_s[ch].append(g['gain'])
            else:
                channel_gains_q[ch].append(g['gain'])

    # === Report ===
    print()
    print("=" * 78)
    print(" D13: LPC PREDICTION GAIN")
    print(f" LPC order={LPC_ORDER}, HP filter 0.5 Hz, window={WINDOW_SIZE} samples")
    print("=" * 78)
    print()

    # Per-patient table
    print(f"{'Patient':<10} {'N_quiet':>8} {'N_seiz':>8} {'Gain_all':>10} {'Gain_quiet':>12} {'Gain_seiz':>11}")
    print("-" * 62)

    all_quiescent_gains = []
    all_seizure_gains = []
    all_gains_flat = []

    for patient in patients_sorted:
        if patient not in summary:
            print(f"{patient:<10} {'N/A':>8}")
            continue
        s = summary[patient]
        all_gains_flat.append(s['mean_all'])
        if not np.isnan(s['mean_quiescent']):
            all_quiescent_gains.append(s['mean_quiescent'])
        if not np.isnan(s['mean_seizure']):
            all_seizure_gains.append(s['mean_seizure'])

        seiz_str = f"{s['mean_seizure']:>10.2f} dB" if not np.isnan(s['mean_seizure']) else f"{'N/A':>11}"
        print(f"{patient:<10} {s['n_quiescent']:>8} {s['n_seizure']:>8} "
              f"{s['mean_all']:>9.2f} dB  {s['mean_quiescent']:>10.2f} dB  {seiz_str}")

    print("-" * 62)
    global_all = np.mean(all_gains_flat) if all_gains_flat else float('nan')
    global_q = np.mean(all_quiescent_gains) if all_quiescent_gains else float('nan')
    global_s = np.mean(all_seizure_gains) if all_seizure_gains else float('nan')
    seiz_global_str = f"{global_s:>10.2f} dB" if not np.isnan(global_s) else f"{'N/A':>11}"
    print(f"{'MEAN':<10} {'':>8} {'':>8} "
          f"{global_all:>9.2f} dB  {global_q:>10.2f} dB  {seiz_global_str}")
    print()

    # Per-channel table (compact)
    print(f"  Per-channel mean prediction gain (all patients pooled):")
    print(f"  {'Ch':>4}  {'Gain_all':>10}  {'Gain_quiet':>12}  {'Gain_seiz':>11}")
    print(f"  {'-'*44}")
    ch_means_q = []      # (channel_idx, mean_gain_dB) for active channels only
    for ch in range(21):
        g_all = np.mean(channel_gains[ch]) if channel_gains[ch] else None
        g_q = np.mean(channel_gains_q[ch]) if channel_gains_q[ch] else None
        g_s = np.mean(channel_gains_s[ch]) if channel_gains_s[ch] else None

        if g_q is None:
            # Dead/flat channel across all patients — mark clearly
            print(f"  {ch:>4}  {'DEAD':>10}   {'DEAD':>11}    {'DEAD':>10}")
            continue

        ch_means_q.append((ch, g_q))
        g_all_str = f"{g_all:>9.2f} dB" if g_all is not None else f"{'N/A':>10}"
        seiz_ch = f"{g_s:>10.2f} dB" if g_s is not None else f"{'N/A':>11}"
        print(f"  {ch:>4}  {g_all_str}  {g_q:>11.2f} dB  {seiz_ch}")

    print()
    if ch_means_q:
        best_ch, best_val = max(ch_means_q, key=lambda x: x[1])
        worst_ch, worst_val = min(ch_means_q, key=lambda x: x[1])
        print(f"  Best  quiescent channel: ch{best_ch:02d}  ({best_val:.2f} dB)")
        print(f"  Worst quiescent channel: ch{worst_ch:02d}  ({worst_val:.2f} dB)")
    print()

    # Verdict
    passed = (not np.isnan(global_q)) and (global_q > PASS_THRESHOLD_DB)

    if passed:
        verdict = "PASS"
        if not np.isnan(global_s) and global_s < global_q - 2.0:
            note = (
                f"Notably, seizure gain ({global_s:.2f} dB) is significantly lower than "
                f"quiescent ({global_q:.2f} dB), consistent with seizure epochs having "
                f"higher-entropy, less auto-correlated waveforms."
            )
        else:
            note = (
                f"Gain is consistent across seizure and quiescent epochs. "
                f"LPC coefficients are stable across brain states."
            )
        interpretation = (
            f"Quiescent gain {global_q:.2f} dB > {PASS_THRESHOLD_DB:.1f} dB threshold. "
            f"LPC is reducing residual entropy enough to justify its 3.1 ms "
            f"latency cost. The prediction filter is capturing EEG autocorrelation "
            f"structure. {note}"
        )
    elif np.isnan(global_q):
        verdict = "SKIP"
        interpretation = (
            "No quiescent windows found. All loaded windows contain seizure activity. "
            "Cannot evaluate quiescent prediction gain."
        )
    else:
        verdict = "FAIL"
        interpretation = (
            f"Quiescent gain {global_q:.2f} dB <= {PASS_THRESHOLD_DB:.1f} dB threshold. "
            f"LPC is NOT providing sufficient prediction gain to justify its latency. "
            f"Possible causes: (1) HP filter removes the long-range correlations LPC "
            f"exploits, (2) 8th order is insufficient for this EEG, "
            f"(3) 10-second windows average over too much non-stationarity. "
            f"Consider: removing LPC stage, increasing order, or shortening autocorr_len."
        )

    print(f"  VERDICT: {verdict}")
    print(f"  {interpretation}")
    print()
    print("=" * 78)

    return {
        'passed': passed,
        'verdict': verdict,
        'global_gain_all_dB': float(global_all),
        'global_gain_quiescent_dB': float(global_q),
        'global_gain_seizure_dB': float(global_s),
        'per_patient': {
            p: {
                'gain_all_dB': float(summary[p]['mean_all']),
                'gain_quiescent_dB': float(summary[p]['mean_quiescent']),
                'gain_seizure_dB': float(summary[p]['mean_seizure']),
            }
            for p in patients_sorted if p in summary
        },
        'per_channel_quiescent_dB': {
            ch: float(np.mean(channel_gains_q[ch])) if channel_gains_q[ch] else float('nan')
            for ch in range(21)
        },
    }


if __name__ == '__main__':
    result = run()
    if result is not None and result['verdict'] != 'SKIP':
        sys.exit(0 if result['passed'] else 1)
