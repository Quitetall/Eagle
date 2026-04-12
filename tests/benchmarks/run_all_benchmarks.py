#!/usr/bin/env python3
"""
LamQuant Gen 6 — Benchmark Gauntlet (FIXED)
============================================
Runs all 4 audits in sequence. Each must pass independently.
"""
import sys
import os
import time
import numpy as np

# Ensure sibling benchmark modules are importable by bare name
sys.path.insert(0, os.path.dirname(__file__))

import benchmark_tnn_memory
import benchmark_biological_fidelity
import benchmark_c_parity


def render_banner(title):
    print("\n" + "=" * 80)
    print(f" {title.center(78)} ")
    print("=" * 80)


def render_result(item, status="PASS"):
    pad = max(1, 65 - len(item))
    colors = {
        "PASS": "\033[92m",  # green
        "FAIL": "\033[91m",  # red
        "SKIP": "\033[93m",  # yellow
    }
    color = colors.get(status, "\033[91m")
    reset = "\033[0m"
    print(f"[*] {item}" + "." * pad + f"[{color}{status}{reset}]")


def main():
    print("\n" + "x" * 80)
    print(" LAMQUANT GEN 6 : BENCHMARK GAUNTLET ".center(80))
    print("x" * 80)

    # 1. MEMORY
    render_banner("AUDIT 1: TNN MEMORY FOOTPRINT")
    try:
        result = benchmark_tnn_memory.run()
        if result is None:
            render_result("TNN Memory (missing ckpt)", "SKIP")
        else:
            used, budget = result
            render_result(f"TNN: {used} / {budget} bytes")
    except SystemExit:
        render_result("TNN Memory", "FAIL")
    except Exception as e:
        print(f"\n[!!!] MEMORY BENCHMARK CRASHED: {e}")
        render_result("TNN Memory", "FAIL")

    time.sleep(0.3)

    # 2. BIOLOGICAL FIDELITY
    render_banner("AUDIT 2: BIOLOGICAL FIDELITY (Held-Out Patients)")
    try:
        result = benchmark_biological_fidelity.run()
        if result is None:
            render_result("Biological Fidelity (missing ckpt or patient data)", "SKIP")
        else:
            _, min_r = result
            render_result(f"Min R = {min_r:.4f} (floor: 0.85)")
    except SystemExit:
        render_result("Biological Fidelity", "FAIL")
    except Exception as e:
        print(f"\n[!!!] FIDELITY BENCHMARK CRASHED: {e}")
        render_result("Biological Fidelity", "FAIL")

    time.sleep(0.3)

    # 3. C PARITY
    render_banner("AUDIT 3: C-SIMULATION BIT PARITY")
    try:
        result = benchmark_c_parity.run()
        if result is None:
            render_result("C Parity (missing ckpt or firmware header)", "SKIP")
        else:
            render_result(f"Max cascaded drift: {result:.6f}")
    except SystemExit:
        render_result("C Parity", "FAIL")
    except Exception as e:
        print(f"\n[!!!] PARITY BENCHMARK CRASHED: {e}")
        render_result("C Parity", "FAIL")

    # 4. CLINICAL HARNESS (requires full dataset + teacher checkpoint)
    render_banner("AUDIT 4: CLINICAL STRESS PROFILES")
    try:
        # clinical_master_harness.py should be in the same directory as this script
        from clinical_master_harness import ClinicalMasterHarness, STRESS_PROFILES
        import torch
        harness = ClinicalMasterHarness(use_gpu=torch.cuda.is_available(), stdout=False)
        harness.run_suite()
        if harness.skipped:
            render_result(f"Clinical Harness ({harness.skip_reason})", "SKIP")
        else:
            pass_count = sum(1 for r in harness.full_results if r['passed'])
            total = len(harness.full_results)
            if pass_count == total:
                render_result(f"Clinical: {pass_count}/{total} profiles passed")
            else:
                render_result(f"Clinical: {pass_count}/{total} profiles passed", "FAIL")
    except ImportError as e:
        print(f"[!] Clinical harness import failed: {e}")
        print(f"[!] Ensure clinical_master_harness.py is in: {os.path.dirname(__file__)}")
        render_result("Clinical Harness (import failed)", "SKIP")
    except Exception as e:
        print(f"[!] Clinical harness error: {e}")
        import traceback
        traceback.print_exc()
        render_result("Clinical Harness (CRASHED)", "FAIL")

    # 5. DEPLOYMENT PATH (ternary encoder → FP32 decoder)
    render_banner("AUDIT 5: DEPLOYMENT PATH (Ternary Encoder → FP32 Decoder)")
    try:
        import benchmark_deployment_path
        result = benchmark_deployment_path.run()
        if result is None:
            render_result("Deployment Path (missing ckpt or patient data)", "SKIP")
        else:
            route_a_results, _ = result
            deploy_rs = [r['r'] for r in route_a_results]
            render_result(f"Deployment Min R = {min(deploy_rs):.4f} (floor: 0.85)")
    except SystemExit:
        render_result("Deployment Path", "FAIL")
    except ImportError as e:
        print(f"[!] Deployment benchmark import failed: {e}")
        render_result("Deployment Path (import failed)", "SKIP")
    except Exception as e:
        print(f"[!] Deployment benchmark error: {e}")
        import traceback
        traceback.print_exc()
        render_result("Deployment Path (CRASHED)", "FAIL")

    # 6. COMPRESSION RATIO (real FSQ + rANS byte count at multiple levels)
    render_banner("AUDIT 6: COMPRESSION RATIO (FSQ + rANS)")
    try:
        import benchmark_compression_ratio
        result = benchmark_compression_ratio.run()
        if result is None:
            render_result("Compression Ratio (missing ckpts or patient data)", "SKIP")
        else:
            # benchmark_compression_ratio.run() returns (golden_results_by_level, lightning_results)
            results_by_level = result[0] if isinstance(result, tuple) else result
            best = None
            for L in sorted(results_by_level.keys()):
                res = results_by_level[L]
                mcr = np.mean([r['cr'] for r in res])
                mr = np.mean([r['r'] for r in res])
                if mr > 0.95:
                    best = (L, mcr, mr)
                    break
            if best:
                L, mcr, mr = best
                render_result(f"Best: L={L}, CR={mcr:.1f}x, R={mr:.4f}")
            else:
                render_result("No level with R > 0.95", "FAIL")
    except SystemExit:
        render_result("Compression Ratio", "FAIL")
    except ImportError as e:
        print(f"[!] Compression ratio import failed: {e}")
        render_result("Compression Ratio (import failed)", "SKIP")
    except Exception as e:
        print(f"[!] Compression ratio error: {e}")
        import traceback
        traceback.print_exc()
        render_result("Compression Ratio (CRASHED)", "FAIL")

    # 7. SUBBAND BOUNDARY LEAKAGE (per-frequency reconstruction error)
    render_banner("AUDIT 7: SUBBAND BOUNDARY LEAKAGE DIAGNOSTIC")
    try:
        import benchmark_subband_leakage
        result = benchmark_subband_leakage.run()
        if result is None:
            render_result("Subband Leakage (missing ckpt or patient data)", "SKIP")
        else:
            diag = result['diagnosis']
            deficit = result['snr_deficit_dB']
            if diag == 'UNIFORM':
                render_result(f"Boundary deficit: {deficit:.1f} dB — {diag}")
            elif diag == 'BORDERLINE':
                render_result(f"Boundary deficit: {deficit:.1f} dB — {diag}", "SKIP")
            else:
                render_result(f"Boundary deficit: {deficit:.1f} dB — {diag}", "FAIL")
    except SystemExit:
        render_result("Subband Leakage", "FAIL")
    except ImportError as e:
        print(f"[!] Subband leakage import failed: {e}")
        render_result("Subband Leakage (import failed)", "SKIP")
    except Exception as e:
        print(f"[!] Subband leakage error: {e}")
        import traceback
        traceback.print_exc()
        render_result("Subband Leakage (CRASHED)", "FAIL")

    # 8. FULLBAND ERROR HEATMAP
    render_banner("AUDIT 8: FULLBAND ERROR HEATMAP (D1)")
    try:
        import benchmark_fullband_error_heatmap
        result = benchmark_fullband_error_heatmap.run()
        if result is None:
            render_result("Fullband Error Heatmap (missing deps)", "SKIP")
        else:
            deficit = result['snr_deficit_dB']
            if result.get('passed', False):
                render_result(f"Boundary SNR deficit: {deficit:.2f} dB < 6 dB")
            else:
                render_result(f"Boundary SNR deficit: {deficit:.2f} dB >= 6 dB", "FAIL")
    except SystemExit:
        render_result("Fullband Error Heatmap", "FAIL")
    except Exception as e:
        print(f"[!] Error: {e}")
        import traceback
        traceback.print_exc()
        render_result("Fullband Error Heatmap (CRASHED)", "FAIL")

    # 9. ABLATION MATRIX
    render_banner("AUDIT 9: ABLATION MATRIX (D2)")
    try:
        import benchmark_ablation_matrix
        result = benchmark_ablation_matrix.run()
        if result is None:
            render_result("Ablation Matrix (missing deps)", "SKIP")
        else:
            n_ran = result['n_configs_ran']
            n_total = result['n_configs_total']
            if result.get('passed', False):
                render_result(f"Ablation: {n_ran}/{n_total} configs ran")
            else:
                render_result(f"Ablation: {n_ran}/{n_total} configs ran (need >= 2)", "FAIL")
    except SystemExit:
        render_result("Ablation Matrix", "FAIL")
    except Exception as e:
        print(f"[!] Error: {e}")
        import traceback
        traceback.print_exc()
        render_result("Ablation Matrix (CRASHED)", "FAIL")

    # 10. FSQ ENTROPY VS ACTIVITY
    render_banner("AUDIT 10: FSQ ENTROPY VS ACTIVITY (D3)")
    try:
        import benchmark_fsq_entropy_activity
        result = benchmark_fsq_entropy_activity.run()
        if result is None:
            render_result("FSQ Entropy vs Activity (missing deps)", "SKIP")
        elif result.get('passed') is None:
            render_result(f"FSQ Entropy: {result.get('reason', 'no seizure windows')}", "SKIP")
        else:
            h_seiz = result['h_seizure']
            h_quiet = result['h_quiet']
            if result.get('passed', False):
                render_result(f"H(seizure)={h_seiz:.4f} > H(quiet)={h_quiet:.4f}")
            else:
                render_result(f"H(seizure)={h_seiz:.4f} <= H(quiet)={h_quiet:.4f}", "FAIL")
    except SystemExit:
        render_result("FSQ Entropy vs Activity", "FAIL")
    except Exception as e:
        print(f"[!] Error: {e}")
        import traceback
        traceback.print_exc()
        render_result("FSQ Entropy vs Activity (CRASHED)", "FAIL")

    # 11. RATE-DISTORTION CURVE
    render_banner("AUDIT 11: RATE-DISTORTION CURVE (D4)")
    try:
        import benchmark_rate_distortion
        result = benchmark_rate_distortion.run()
        if result is None:
            render_result("Rate-Distortion Curve (missing deps)", "SKIP")
        else:
            pass_points = result.get('pass_points', [])
            if result.get('passed', False):
                best = max(pass_points, key=lambda x: x[2])
                render_result(f"Best: L={best[0]}, bps={best[1]:.4f}, R={best[2]:.4f}")
            else:
                rd = result.get('rd_points', [])
                best_r = max((r for _, _, r, _, _ in rd), default=0.0)
                render_result(f"No point R>=0.85 at bps<1.5 (best R={best_r:.4f})", "FAIL")
    except SystemExit:
        render_result("Rate-Distortion Curve", "FAIL")
    except Exception as e:
        print(f"[!] Error: {e}")
        import traceback
        traceback.print_exc()
        render_result("Rate-Distortion Curve (CRASHED)", "FAIL")

    # 12. PATIENT SPREAD
    render_banner("AUDIT 12: PATIENT SPREAD (D5)")
    try:
        import benchmark_patient_spread
        result = benchmark_patient_spread.run()
        if result is None:
            render_result("Patient Spread (missing deps)", "SKIP")
        else:
            range_r = result['range_r']
            min_r = result['min_r']
            if result.get('passed', False):
                render_result(f"Range={range_r:.4f} < 0.10, Min R={min_r:.4f} > 0.80")
            else:
                render_result(f"Range={range_r:.4f}, Min R={min_r:.4f}", "FAIL")
    except SystemExit:
        render_result("Patient Spread", "FAIL")
    except Exception as e:
        print(f"[!] Error: {e}")
        import traceback
        traceback.print_exc()
        render_result("Patient Spread (CRASHED)", "FAIL")

    # 13. SEIZURE VS QUIESCENT R
    render_banner("AUDIT 13: SEIZURE VS QUIESCENT R (D6)")
    try:
        import benchmark_seizure_vs_quiescent
        result = benchmark_seizure_vs_quiescent.run()
        if result is None:
            render_result("Seizure vs Quiescent (missing deps)", "SKIP")
        elif result.get('passed') is None:
            render_result(f"Seizure vs Quiescent: {result.get('reason', 'skip')}", "SKIP")
        else:
            delta = result['delta']
            if result.get('passed', False):
                render_result(f"Delta(quie-seiz)={delta:.4f} < 0.15")
            else:
                render_result(f"Delta(quie-seiz)={delta:.4f} >= 0.15", "FAIL")
    except SystemExit:
        render_result("Seizure vs Quiescent", "FAIL")
    except Exception as e:
        print(f"[!] Error: {e}")
        import traceback
        traceback.print_exc()
        render_result("Seizure vs Quiescent (CRASHED)", "FAIL")

    # 14. CLINICAL PRESERVATION MODE 0
    render_banner("AUDIT 14: CLINICAL PRESERVATION MODE 0 (D7)")
    try:
        import benchmark_clinical_preservation
        result = benchmark_clinical_preservation.run()
        if result is None:
            render_result("Clinical Preservation (missing deps)", "SKIP")
        elif result.get('passed') is None:
            render_result(f"Clinical Preservation: {result.get('reason', 'skip')}", "SKIP")
        else:
            ratio = result['sensitivity_ratio']
            if result.get('passed', False):
                render_result(f"Sensitivity ratio={ratio:.3f} >= 0.80")
            else:
                render_result(f"Sensitivity ratio={ratio:.3f} < 0.80", "FAIL")
    except SystemExit:
        render_result("Clinical Preservation Mode 0", "FAIL")
    except Exception as e:
        print(f"[!] Error: {e}")
        import traceback
        traceback.print_exc()
        render_result("Clinical Preservation Mode 0 (CRASHED)", "FAIL")

    # 15. SPECTRAL FIDELITY PSD
    render_banner("AUDIT 15: SPECTRAL FIDELITY PSD (D8)")
    try:
        import benchmark_spectral_fidelity
        result = benchmark_spectral_fidelity.run()
        if result is None:
            render_result("Spectral Fidelity PSD (missing deps)", "SKIP")
        else:
            f_3db = result['f_3dB_hz']
            if result.get('passed', False):
                render_result(f"f_3dB={f_3db:.1f} Hz > 30 Hz")
            else:
                render_result(f"f_3dB={f_3db:.1f} Hz <= 30 Hz", "FAIL")
    except SystemExit:
        render_result("Spectral Fidelity PSD", "FAIL")
    except Exception as e:
        print(f"[!] Error: {e}")
        import traceback
        traceback.print_exc()
        render_result("Spectral Fidelity PSD (CRASHED)", "FAIL")

    # 16. LATENT UTILIZATION
    render_banner("AUDIT 16: LATENT UTILIZATION (D9)")
    try:
        import benchmark_latent_utilization
        result = benchmark_latent_utilization.run()
        if result is None:
            render_result("Latent Utilization (missing deps)", "SKIP")
        else:
            mean_util = result['mean_utilization']
            min_util = result['min_utilization']
            if result.get('passed', False):
                render_result(f"Mean util={mean_util:.1f}% > 60%, Min={min_util:.1f}% >= 25%")
            else:
                render_result(f"Mean util={mean_util:.1f}%, Min={min_util:.1f}%", "FAIL")
    except SystemExit:
        render_result("Latent Utilization", "FAIL")
    except Exception as e:
        print(f"[!] Error: {e}")
        import traceback
        traceback.print_exc()
        render_result("Latent Utilization (CRASHED)", "FAIL")

    # 17. XNOR+CPOP KERNEL
    render_banner("AUDIT 17: XNOR+CPOP KERNEL (D10)")
    try:
        import benchmark_xnor_cpop_stub
        result = benchmark_xnor_cpop_stub.run()
        if result is None:
            render_result("XNOR+cpop Kernel (missing deps)", "SKIP")
        else:
            wall_ms = result['wall_ideal_ms']
            margin = result['real_time_margin']
            if result.get('passed', False):
                render_result(f"Ideal wall={wall_ms:.2f} ms, RT margin={margin:.1f}x")
            else:
                render_result(f"XNOR+cpop Kernel", "FAIL")
    except SystemExit:
        render_result("XNOR+cpop Kernel", "FAIL")
    except Exception as e:
        print(f"[!] Error: {e}")
        import traceback
        traceback.print_exc()
        render_result("XNOR+cpop Kernel (CRASHED)", "FAIL")

    # 18. E2E LATENCY
    render_banner("AUDIT 18: E2E LATENCY (D11)")
    try:
        import benchmark_e2e_latency_stub
        result = benchmark_e2e_latency_stub.run()
        if result is None:
            render_result("E2E Latency (missing deps)", "SKIP")
        else:
            total_ms = result['total_ms']
            margin = result['real_time_margin']
            if result.get('passed', False):
                render_result(f"Total={total_ms:.2f} ms, RT margin={margin:.1f}x")
            else:
                render_result(f"Total={total_ms:.2f} ms (not real-time)", "FAIL")
    except SystemExit:
        render_result("E2E Latency", "FAIL")
    except Exception as e:
        print(f"[!] Error: {e}")
        import traceback
        traceback.print_exc()
        render_result("E2E Latency (CRASHED)", "FAIL")

    # 19. DETAIL SPARSITY
    render_banner("AUDIT 19: DETAIL SPARSITY (D12)")
    try:
        import benchmark_detail_sparsity
        result = benchmark_detail_sparsity.run()
        if result is None:
            render_result("Detail Sparsity (missing deps)", "SKIP")
        else:
            l1_sparsity = result['l1_sparsity_mean']
            if result.get('passed', False):
                render_result(f"L1 mean sparsity={l1_sparsity*100:.1f}% > threshold")
            else:
                render_result(f"L1 mean sparsity={l1_sparsity*100:.1f}% below threshold", "FAIL")
    except SystemExit:
        render_result("Detail Sparsity", "FAIL")
    except Exception as e:
        print(f"[!] Error: {e}")
        import traceback
        traceback.print_exc()
        render_result("Detail Sparsity (CRASHED)", "FAIL")

    # 20. LPC GAIN
    render_banner("AUDIT 20: LPC GAIN (D13)")
    try:
        import benchmark_lpc_gain
        result = benchmark_lpc_gain.run()
        if result is None:
            render_result("LPC Gain (missing deps)", "SKIP")
        else:
            gain_db = result['global_gain_quiescent_dB']
            if result.get('passed', False):
                render_result(f"Global quiescent gain={gain_db:.2f} dB")
            else:
                render_result(f"Global quiescent gain={gain_db:.2f} dB (below threshold)", "FAIL")
    except SystemExit:
        render_result("LPC Gain", "FAIL")
    except Exception as e:
        print(f"[!] Error: {e}")
        import traceback
        traceback.print_exc()
        render_result("LPC Gain (CRASHED)", "FAIL")

    # 21. CAYLEY ROTATION
    render_banner("AUDIT 21: CAYLEY ROTATION (D14)")
    try:
        import benchmark_cayley_rotation
        result = benchmark_cayley_rotation.run()
        if result is None:
            render_result("Cayley Rotation (missing deps)", "SKIP")
        else:
            delta = result['entropy_delta']
            if result.get('passed', False):
                render_result(f"Entropy delta={delta:.4f} (rotation improves utilization)")
            else:
                render_result(f"Entropy delta={delta:.4f} (rotation did not help)", "FAIL")
    except SystemExit:
        render_result("Cayley Rotation", "FAIL")
    except Exception as e:
        print(f"[!] Error: {e}")
        import traceback
        traceback.print_exc()
        render_result("Cayley Rotation (CRASHED)", "FAIL")

    # 22. FSQ VALIDATION
    render_banner("AUDIT 22: FSQ VALIDATION (D15)")
    try:
        import benchmark_fsq_validation
        result = benchmark_fsq_validation.run()
        if result is None:
            render_result("FSQ Validation (missing deps)", "SKIP")
        else:
            our_r = result['our_fsq_r']
            lr_r = result['lucidrains_fsq_r']
            delta = result['r_delta']
            if result.get('passed', False):
                render_result(f"Ours R={our_r:.4f}, lucidrains R={lr_r:.4f}, delta={delta:.4f}")
            else:
                render_result(f"Ours R={our_r:.4f}, lucidrains R={lr_r:.4f}, delta={delta:.4f}", "FAIL")
    except SystemExit:
        render_result("FSQ Validation", "FAIL")
    except ImportError as e:
        print(f"[!] FSQ validation import failed: {e}")
        render_result("FSQ Validation (import failed)", "SKIP")
    except Exception as e:
        print(f"[!] FSQ validation error: {e}")
        import traceback
        traceback.print_exc()
        render_result("FSQ Validation (CRASHED)", "FAIL")

    print("\n" + "*" * 80)
    print(" GAUNTLET COMPLETE ".center(80))
    print("*" * 80 + "\n")


if __name__ == "__main__":
    main()
