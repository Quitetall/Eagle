#!/usr/bin/env python3
"""
LamQuant Gen 7.5 — D11: E2E Pipeline Latency Profile (Hazard3 stub)
====================================================================
Diagnostic D11: Where is the actual latency bottleneck?

This is a theoretical-only analysis.  No hardware is required.

Models the complete on-device pipeline for one 10-second EEG window:

  Stage 1: Biquad HP filter        — 21 channels × 2500 samples × 5 cycles/sample
  Stage 2: LPC analysis            — 21 ch × (256 autocorr × 8-order + 36 Levinson steps)
  Stage 3: Lifting DWT (3 levels)  — 21 ch × (2500+1250+625) samples × 4 cycles
  Stage 4: TNN encoder             — from D10 MAC count × (5/16)
  Stage 5: FSQ quantize            — 32 × 79 × 10 cycles
  Stage 6: rANS encode             — 32 × 79 × 15 cycles
  Stage 7: Detail subband encode   — estimated from ~80% sparsity
  Stage 8: BLE packet assembly     — ~5000 cycles fixed overhead

All cycle counts are per 10-second window at 250 Hz (2500 samples/channel).
Wall time = total_cycles / 150 MHz.
Real-time margin = 10.0 s / wall_time_s.

PASS criterion: total wall time < 10 seconds (real-time capable).

Usage:
  python benchmark_e2e_latency_stub.py
  python benchmark_e2e_latency_stub.py --checkpoint path/to/model.ckpt
"""

import os
import sys
import numpy as np
from pathlib import Path


def find_project_root(marker='.git'):
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / marker).exists():
            return str(parent)
    raise RuntimeError("Project root not found")


ROOT_DIR = find_project_root()
sys.path.append(os.path.join(ROOT_DIR, 'ai_models', 'student'))

CPU_FREQ_HZ   = 150e6    # 150 MHz Hazard3 on RP2350
WINDOW_SEC    = 10.0     # one acquisition window
N_CH          = 21       # EEG channels
FS            = 250      # sampling rate (Hz)
T_RAW         = int(WINDOW_SEC * FS)    # 2500 raw samples per channel

# TNN encoder parameters (Gen 7.5 canonical)
TNN_MACS_PER_CYCLE = 16 / 5    # XNOR+cpop pipeline (D10)
TNN_OVERHEAD_FACTOR = 1.30     # +30% realistic overhead

# Latent dimensions
LATENT_CH  = 32
LATENT_T   = 79

# Detail subband sparsity (from D12 benchmark: ~80% of coefficients below threshold)
DETAIL_SPARSITY = 0.80


def get_tnn_macs(width=96):
    """Return total encoder MACs for Gen 7.5 at given width W.
    Mirrors the computation in benchmark_xnor_cpop_stub.py.
    """
    W      = width
    in_ch  = N_CH
    latent = LATENT_CH

    T_in      = 313           # L3 approximation length
    T_focal1  = (T_in + 1) // 2     # 157
    T_focal2  = T_focal1             # 157 (stride 1)
    T_focal3  = (T_focal2 + 1) // 2  # 79
    T_lat     = T_focal3

    def cmacs(ic, oc, k, ol, groups=1):
        return int(oc * (ic / groups) * k * ol)

    macs = (
        cmacs(in_ch, in_ch, 1, T_in)       +  # premix
        cmacs(in_ch, W, 3, T_focal1)        +  # focal1_conv
        0                                    +  # focal1_sc (zero-pad)
        cmacs(W, W, 5, T_focal2)            +  # focal2.conv
        0                                    +  # focal2.sc (identity)
        cmacs(W, W, 7, T_focal3)            +  # focal3.conv
        cmacs(W, W, 1, T_focal3)            +  # focal3.sc (strided 1x1)
        cmacs(W, W, 3, T_lat, groups=W)     +  # dw_gate (depthwise)
        cmacs(W, latent, 1, T_lat)          +  # bneck_v
        cmacs(W, latent, 1, T_lat)             # bneck_g
    )
    return macs


def build_pipeline_stages(width=96):
    """Build ordered list of pipeline stages with their cycle estimates.

    Returns list of (stage_name, cycles, description) tuples.
    """
    # Stage 1: Biquad HP filter
    # 2nd-order Butterworth at 0.5 Hz: 5 multiplies + 4 adds = ~5 cycles/sample
    hp_cycles = N_CH * T_RAW * 5

    # Stage 2: LPC analysis
    # Per channel: 256-point autocorrelation for order 8 (256 × 8 MACs)
    #              Levinson-Durbin for order 8 (~36 operations)
    lpc_autocorr = N_CH * (256 * 8)     # autocorrelation MACs
    lpc_levinson = N_CH * 36            # Levinson steps
    lpc_cycles   = lpc_autocorr + lpc_levinson

    # Stage 3: Lifting DWT (3 levels, Le Gall 5/3)
    # Level 1: T_RAW samples → T_RAW/2 + T_RAW/2 = T_RAW (both approx + detail)
    # Level 2: T_RAW/2 approximation
    # Level 3: T_RAW/4 approximation
    # Per sample per channel: predict + update = 4 multiplications
    dwt_samples  = T_RAW + T_RAW // 2 + T_RAW // 4   # 2500 + 1250 + 625 = 4375
    dwt_cycles   = N_CH * dwt_samples * 4

    # Stage 4: TNN encoder
    tnn_macs   = get_tnn_macs(width)
    tnn_cycles = int(tnn_macs / TNN_MACS_PER_CYCLE * TNN_OVERHEAD_FACTOR)

    # Stage 5: FSQ quantize
    # Per symbol: find min/max, scale, clip, round → ~10 cycles
    fsq_cycles = LATENT_CH * LATENT_T * 10

    # Stage 6: rANS encode
    # Per symbol: one rANS state update → ~15 cycles (division-heavy)
    rans_cycles = LATENT_CH * LATENT_T * 15

    # Stage 7: Detail subband encode
    # L1 detail: T_RAW/2 = 1250 coefficients per channel (21 ch)
    # L2 detail: T_RAW/4 = 625 per channel
    # L3 detail: T_RAW/8 = 312 per channel
    # With 80% sparsity: only 20% are encoded.
    # Non-zero coefficients use Golomb-Rice: ~10 cycles/coeff + run-length overhead
    detail_total   = N_CH * (T_RAW // 2 + T_RAW // 4 + T_RAW // 8)
    detail_nonzero = int(detail_total * (1.0 - DETAIL_SPARSITY))
    detail_cycles  = detail_nonzero * 10 + detail_total // 4   # RLE + encoding

    # Stage 8: BLE packet assembly
    ble_cycles = 5000   # header framing, CRC, fragmentation

    stages = [
        ('Biquad HP filter',       hp_cycles,      f'{N_CH} ch × {T_RAW} samp × 5 cyc/samp'),
        ('LPC analysis',           lpc_cycles,      f'{N_CH} ch × ({256}×8 autocorr + 36 Levinson)'),
        ('Lifting DWT (3 levels)', dwt_cycles,      f'{N_CH} ch × {dwt_samples} samp × 4 cyc'),
        ('TNN encoder',            tnn_cycles,      f'{tnn_macs:,} MACs × {TNN_OVERHEAD_FACTOR:.2f} overhead'),
        ('FSQ quantize',           fsq_cycles,      f'{LATENT_CH}×{LATENT_T} × 10 cyc'),
        ('rANS encode',            rans_cycles,     f'{LATENT_CH}×{LATENT_T} × 15 cyc'),
        ('Detail subband encode',  detail_cycles,   f'{detail_nonzero} non-zero ({DETAIL_SPARSITY*100:.0f}% sparse)'),
        ('BLE packet assembly',    ble_cycles,      'header + CRC + fragmentation'),
    ]
    return stages


def run(checkpoint_path=None):
    # Checkpoint discovery — only used to read width W; no model loaded
    width = 96  # Gen 7.5 canonical default
    width_source = "default (Gen 7.5 canonical)"

    if checkpoint_path is None:
        for candidate in [
            os.path.join(ROOT_DIR, 'ai_models/student/student_hardened.ckpt'),
            os.path.join(ROOT_DIR, 'weights/student_subband.ckpt'),
            os.path.join(ROOT_DIR, 'weights/student_subband_fast.ckpt'),
        ]:
            if os.path.exists(candidate):
                checkpoint_path = candidate
                break

    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            import torch
            sd = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            if 'model_state_dict' in sd:
                sd = sd['model_state_dict']
            if 'focal2.conv.weight' in sd:
                width = sd['focal2.conv.weight'].shape[0]
                width_source = f"checkpoint ({checkpoint_path})"
        except Exception as e:
            print(f"[!] Could not read width from checkpoint: {e}")

    stages = build_pipeline_stages(width)
    total_cycles = sum(cyc for _, cyc, _ in stages)
    total_ms     = total_cycles / CPU_FREQ_HZ * 1e3
    total_sec    = total_cycles / CPU_FREQ_HZ
    margin       = WINDOW_SEC / total_sec

    print(f"[*] D11: E2E Pipeline Latency Profile (Hazard3 stub)")
    print(f"    Width source : {width_source}")
    print(f"    Width W      : {width}")
    print(f"    Window       : {WINDOW_SEC:.1f} s ({T_RAW} samples @ {FS} Hz)")
    print(f"    CPU freq     : {CPU_FREQ_HZ/1e6:.0f} MHz Hazard3")
    print(f"    Det. sparsity: {DETAIL_SPARSITY*100:.0f}%")
    print()

    print("=" * 90)
    print(" D11: E2E PIPELINE LATENCY PROFILE — Per-stage cycle breakdown")
    print("=" * 90)
    print(f"  {'Stage':<28} {'Cycles':>14} {'ms':>8} {'% Total':>9}   Notes")
    print(f"  {'-'*80}")

    for name, cyc, notes in stages:
        pct = cyc / total_cycles * 100
        ms_val = cyc / CPU_FREQ_HZ * 1e3
        print(f"  {name:<28} {cyc:>14,} {ms_val:>8.2f} {pct:>8.1f}%   {notes}")

    print(f"  {'-'*80}")
    print(f"  {'TOTAL':<28} {total_cycles:>14,} {total_ms:>8.2f} {'100.0%':>9}")
    print()

    print("=" * 90)
    print(" REAL-TIME ANALYSIS")
    print("=" * 90)
    print(f"  Total cycles          : {total_cycles:>14,}")
    print(f"  Wall time             : {total_ms:>14.2f} ms  ({total_sec:.4f} s)")
    print(f"  Window duration       : {WINDOW_SEC*1e3:>14.0f} ms  ({WINDOW_SEC:.1f} s)")
    print(f"  Real-time margin      : {margin:>14.1f}x")
    print()

    # Identify bottleneck (stage with most cycles)
    bottleneck_name, bottleneck_cyc, _ = max(stages, key=lambda s: s[1])
    bottleneck_pct = bottleneck_cyc / total_cycles * 100
    print(f"  Bottleneck stage      : {bottleneck_name} ({bottleneck_pct:.1f}% of total)")
    print()

    print("=" * 90)
    print(" LATENCY REDUCTION OPPORTUNITIES")
    print("=" * 90)
    print()
    print("  1. TNN encoder: Use XNOR+cpop packing to achieve ideal 16 MACs/5 cycles.")
    print("     Current estimate includes +30% overhead; firmware optimization can reduce.")
    print()
    print("  2. Biquad HP filter: SIMD vectorise using P-extension or Zbe extension.")
    print("     21-channel biquad can be parallelised 4-way → 4x throughput.")
    print()
    print("  3. Detail subband encode: Increase sparsity threshold to reduce non-zeros.")
    print("     90% sparsity (vs current 80%) halves the detail-encode cost.")
    print()
    print("  4. rANS encode: Replace division-heavy rANS with ANS table lookup (TANS).")
    print("     Pre-built 16-entry table → ~5 cycles/symbol vs current 15.")
    print()
    print("  5. DMA overlap: Stream BLE packet while encoding the next detail subband.")
    print("     Overlapping stages 6+7+8 can hide ~8% of total latency.")
    print()

    # Pass/fail
    passed = total_sec < WINDOW_SEC

    if passed:
        print(f"  [PASS] Total {total_ms:.2f} ms < {WINDOW_SEC*1e3:.0f} ms window — real-time capable ({margin:.1f}x margin)")
    else:
        print(f"  [FAIL] Total {total_ms:.2f} ms > {WINDOW_SEC*1e3:.0f} ms window — NOT real-time capable")
        print(f"         Need {total_sec/WINDOW_SEC:.2f}x faster to meet real-time constraint.")
    print()
    print("=" * 90)

    return {
        'passed': passed,
        'total_cycles': total_cycles,
        'total_ms': total_ms,
        'real_time_margin': margin,
        'bottleneck': bottleneck_name,
        'stages': [(n, c) for n, c, _ in stages],
        'width': width,
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=None)
    args = parser.parse_args()
    result = run(args.checkpoint)
    if result:
        sys.exit(0 if result['passed'] else 1)
