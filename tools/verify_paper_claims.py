#!/usr/bin/env python3
"""Cross-check every measured claim in the LamQuant Lossless paper
against the JSON evidence files that were used to produce them.

Each entry below is (claim_text, source_path, computation_or_value,
expected, tolerance). Prints PASS / MISMATCH per claim, summary
at the end.

Usage:
  python3 tools/verify_paper_claims.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "outputs" / "paper"


def load(p: Path) -> Any:
    with open(p) as f:
        return json.load(f)


def approx_eq(a: float, b: float, tol: float = 0.001) -> bool:
    if a == 0 and b == 0:
        return True
    return abs(a - b) / max(abs(b), 1e-12) <= tol


def main() -> int:
    fails: list[str] = []
    passes: list[str] = []

    def check(name: str, ok: bool, detail: str) -> None:
        prefix = "PASS" if ok else "FAIL"
        line = f"[{prefix}] {name} — {detail}"
        (passes if ok else fails).append(line)
        print(line, flush=True)

    # ---- CHB-MIT claims ----
    chbmit = load(OUT / "chbmit_lpc_mode_compare.json")
    adaptive = next(r for r in chbmit["results"] if r["mode"] == "adaptive")
    check("CHB-MIT files=686",
          adaptive["files_in"] == 686,
          f"json={adaptive['files_in']}")
    check("CHB-MIT raw=45,756,994,304 B (45.76 GB)",
          adaptive["input_bytes"] == 45756994304,
          f"json={adaptive['input_bytes']}")
    check("CHB-MIT compressed=16,804,264,365 B (16.80 GB)",
          adaptive["output_bytes"] == 16804264365,
          f"json={adaptive['output_bytes']}")
    check("CHB-MIT CR=2.7229:1",
          approx_eq(adaptive["cr"], 2.7229, 0.0001),
          f"json={adaptive['cr']:.4f}")
    check("CHB-MIT GB → 45.76 (decimal)",
          approx_eq(45756994304 / 1e9, 45.76, 0.001),
          f"calc={45756994304 / 1e9:.4f}")
    check("CHB-MIT GB → 16.80 (decimal)",
          approx_eq(16804264365 / 1e9, 16.80, 0.001),
          f"calc={16804264365 / 1e9:.4f}")
    check("Improvement 15.9% over Chen 2.35",
          approx_eq((2.7229 - 2.35) / 2.35 * 100, 15.9, 0.05),
          f"calc={(2.7229 - 2.35) / 2.35 * 100:.2f}%")

    # ---- gzip baseline ----
    gz = load(OUT / "gzip_baseline_000.json")
    check("gzip files=453",
          gz["files"] == 453, f"json={gz['files']}")
    check("gzip CR=1.6033",
          approx_eq(gz["cr"], 1.6033, 0.001), f"json={gz['cr']:.4f}")
    check("gzip raw 6.01 GB",
          approx_eq(gz["input_bytes"] / 1e9, 6.01, 0.005),
          f"calc={gz['input_bytes'] / 1e9:.4f}")
    check("gzip compressed 3.75 GB",
          approx_eq(gz["output_bytes"] / 1e9, 3.75, 0.005),
          f"calc={gz['output_bytes'] / 1e9:.4f}")
    check("gzip wall ≈ 271 s",
          approx_eq(gz["wall_seconds"], 271, 0.02),
          f"json={gz['wall_seconds']:.0f}s")

    # ---- TUEG v2.0.2 per-subset (headline) ----
    tueg = load(OUT / "tueg_subset_breakdown_montage.json")
    assert "v2.0.2" in tueg["tree"], (
        f"expected v2.0.2 tree in JSON; got: {tueg['tree']}")
    check("TUEG v2.0.2 files=70,831 (matches AAREADME)",
          tueg["files_total"] == 70831,
          f"json={tueg['files_total']}")
    tueg_in = sum(g["input_bytes"] for g in tueg["groups"].values())
    tueg_out = sum(g["output_bytes"] for g in tueg["groups"].values())
    tueg_cr = tueg_in / tueg_out
    check("TUEG v2.0.2 sum raw = 1,756,355,590,458 B",
          tueg_in == 1756355590458,
          f"json sum={tueg_in}")
    check("TUEG v2.0.2 sum compressed = 768,043,519,030 B",
          tueg_out == 768043519030,
          f"json sum={tueg_out}")
    check("TUEG v2.0.2 CR = 2.287:1",
          approx_eq(tueg_cr, 2.287, 0.001),
          f"calc={tueg_cr:.5f}")
    check("TUEG v2.0.2 1.76 TB (decimal) headline",
          approx_eq(tueg_in / 1e12, 1.76, 0.005),
          f"calc={tueg_in / 1e12:.4f} TB")

    # Per-montage (v2.0.2)
    for grp, gb_claim_in, gb_claim_out, cr_claim in [
        ("01_tcp_ar", 1373.05, 603.11, 2.277),
        ("02_tcp_le", 249.38, 109.64, 2.274),
        ("03_tcp_ar_a", 133.70, 55.19, 2.422),
        ("04_tcp_le_a", 0.2173, 0.0937, 2.318),
    ]:
        g = tueg["groups"][grp]
        check(f"TUEG[{grp}] in={gb_claim_in} GB",
              approx_eq(g["input_bytes"] / 1e9, gb_claim_in, 0.001),
              f"calc={g['input_bytes'] / 1e9:.4f}")
        check(f"TUEG[{grp}] out={gb_claim_out} GB",
              approx_eq(g["output_bytes"] / 1e9, gb_claim_out, 0.001),
              f"calc={g['output_bytes'] / 1e9:.4f}")
        check(f"TUEG[{grp}] cr={cr_claim}",
              approx_eq(g["cr"], cr_claim, 0.001),
              f"json cr={g['cr']:.4f}")

    # ---- Shannon entropy claims ----
    for corpus, h_raw_c, h_diff_c, cr_raw_c, cr_diff_c in [
        ("tuar", 9.862, 8.327, 1.622, 1.922),
        ("tueg", 10.775, 9.337, 1.485, 1.714),
        ("tuev", 7.063, 5.620, 2.265, 2.847),
        ("tusl", 10.402, 8.327, 1.538, 1.922),
        ("tusz", 10.412, 8.865, 1.537, 1.805),
        ("chbmit", 8.387, 6.724, 1.908, 2.379),
    ]:
        sh = load(OUT / f"shannon_entropy_{corpus}.json")
        check(f"Shannon[{corpus}] H_raw={h_raw_c}",
              approx_eq(sh["H_raw"], h_raw_c, 0.0005),
              f"json={sh['H_raw']:.4f}")
        check(f"Shannon[{corpus}] H_diff={h_diff_c}",
              approx_eq(sh["H_diff"], h_diff_c, 0.0005),
              f"json={sh['H_diff']:.4f}")
        check(f"Shannon[{corpus}] CR_raw_ceil={cr_raw_c}",
              approx_eq(sh["cr_raw_ceiling"], cr_raw_c, 0.001),
              f"json={sh['cr_raw_ceiling']:.4f}")
        check(f"Shannon[{corpus}] CR_diff_ceil={cr_diff_c}",
              approx_eq(sh["cr_diff_ceiling"], cr_diff_c, 0.001),
              f"json={sh['cr_diff_ceiling']:.4f}")

    # Shannon aggregate (paper claims 9.244 H_raw, 7.656 H_diff)
    full = load(OUT / "shannon_entropy_full_summary.json")
    a = full["aggregate"]
    check("Shannon aggregate H_raw=9.244",
          approx_eq(a["H_raw_weighted"], 9.244, 0.0005),
          f"json={a['H_raw_weighted']:.4f}")
    check("Shannon aggregate H_diff=7.656",
          approx_eq(a["H_diff_weighted"], 7.656, 0.0005),
          f"json={a['H_diff_weighted']:.4f}")
    check("Shannon aggregate CR_raw_ceil=1.731",
          approx_eq(a["cr_raw_ceiling"], 1.731, 0.001),
          f"json={a['cr_raw_ceiling']:.4f}")
    check("Shannon aggregate CR_diff_ceil=2.090",
          approx_eq(a["cr_diff_ceiling"], 2.090, 0.001),
          f"json={a['cr_diff_ceiling']:.4f}")

    # ---- Physical claims (§I) — current paper config ----
    # 32 ch × 1024 Hz × 24-bit = 786,432 bps ≈ 786 kbps (TUEG max)
    bps_ambulatory = 32 * 1024 * 24
    check("§I: 32ch×1024Hz×24b ≈ 786 kbps",
          approx_eq(bps_ambulatory / 1e3, 786, 0.001),
          f"calc={bps_ambulatory / 1e3:.0f} kbps")
    # 256 ch × 1 kSPS × 24-bit ≈ 6.3 Mbps (high-density)
    bps_highdens = 256 * 1024 * 24
    check("§I: 256ch×1kSPS×24b ≈ 6.3 Mbps",
          approx_eq(bps_highdens / 1e6, 6.3, 0.01),
          f"calc={bps_highdens / 1e6:.2f} Mbps")
    # 32 ch × 16 kSPS × 24-bit ≈ 12 Mbps (intracranial)
    bps_clinical = 32 * 16000 * 24
    check("§I: 32ch×16kSPS×24b ≈ 12 Mbps",
          approx_eq(bps_clinical / 1e6, 12, 0.05),
          f"calc={bps_clinical / 1e6:.2f} Mbps")

    # ---- Summary ----
    print()
    print(f"=== {len(passes)} PASS / {len(fails)} FAIL ===")
    if fails:
        print("\nFAILURES:")
        for f in fails:
            print(f"  {f}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
