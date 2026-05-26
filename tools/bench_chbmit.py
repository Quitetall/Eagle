#!/usr/bin/env python3
"""CHB-MIT compression-ratio bench across the three LPC modes.

Encodes every EDF under `/mnt/4tb/data/Archive/edf/physionet/chbmit/`
once per LPC mode (`fixed`, `adaptive`, `anytime`), sums input and
output bytes, reports CR per mode + headline best mode.

For the LamQuant TBioCAS paper Table II (Phase 2, item 10).

Output:
    outputs/paper/chbmit_lpc_mode_compare.json
    + stdout report.

Usage:
    python3 tools/bench_chbmit.py
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LML_BIN = REPO_ROOT / "target" / "release" / "lml"
CHBMIT_ROOT = Path("/mnt/4tb/data/Archive/edf/physionet/chbmit")
OUT_DIR = REPO_ROOT / "outputs" / "paper"
LPC_MODES = ("fixed", "adaptive", "anytime")


def ensure_lml_binary() -> None:
    """`cargo build --release --bin lml` if the binary is missing."""
    if LML_BIN.is_file():
        return
    print(f"[bench_chbmit] building {LML_BIN} (release)…", file=sys.stderr)
    subprocess.check_call(
        ["cargo", "build", "--release", "--bin", "lml",
         "--manifest-path", str(REPO_ROOT / "Cargo.toml")],
        cwd=REPO_ROOT,
    )


def find_chbmit_edfs() -> list[Path]:
    """Walk the CHB-MIT tree, return EDF paths (exclude .seizures files)."""
    return sorted(
        p for p in CHBMIT_ROOT.rglob("*.edf")
        if ".seizures" not in p.name
    )


def encode_one(edf: Path, mode: str, scratch: Path) -> int:
    """Encode `edf` under `mode` into `scratch`. Return compressed bytes."""
    # `lml encode <input> -o <output> --bare-lml --lpc-mode <mode>`
    # `--bare-lml` skips the per-file LMA bundle so output size is just the
    # `.lml` codec bytes — apples-to-apples with prior-art CR numbers that
    # report compressed signal bytes only.
    out_path = scratch / (edf.stem + ".lml")
    subprocess.check_call(
        [str(LML_BIN), "encode", str(edf),
         "-o", str(out_path),
         "--bare-lml", "--i-understand-data-loss",
         "--lpc-mode", mode],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    size = out_path.stat().st_size
    out_path.unlink()  # free disk per-file; we only care about size
    return size


def bench_mode(edfs: list[Path], mode: str, scratch: Path) -> dict:
    """Encode all EDFs under one LPC mode, return summary dict."""
    total_in = 0
    total_out = 0
    t0 = time.time()
    failures = 0
    for i, edf in enumerate(edfs):
        try:
            in_sz = edf.stat().st_size
            out_sz = encode_one(edf, mode, scratch)
            total_in += in_sz
            total_out += out_sz
        except subprocess.CalledProcessError as e:
            print(f"[bench_chbmit] {mode}: FAIL on {edf.name}: {e}",
                  file=sys.stderr)
            failures += 1
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            cr = total_in / max(total_out, 1)
            print(f"[bench_chbmit] {mode}: {i+1}/{len(edfs)} "
                  f"({elapsed:.0f}s) CR={cr:.3f} fail={failures}",
                  file=sys.stderr)
    elapsed = time.time() - t0
    return {
        "mode": mode,
        "files_in": len(edfs),
        "files_failed": failures,
        "input_bytes": total_in,
        "output_bytes": total_out,
        "cr": total_in / max(total_out, 1),
        "wall_seconds": elapsed,
        "throughput_mibs": (total_in / max(elapsed, 1)) / (1024 * 1024),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="Encode only the first N EDFs (smoke test).")
    ap.add_argument("--modes", default=",".join(LPC_MODES),
                    help="Comma-separated subset of LPC modes to bench.")
    args = ap.parse_args()

    if not CHBMIT_ROOT.is_dir():
        print(f"[bench_chbmit] missing {CHBMIT_ROOT}", file=sys.stderr)
        return 1

    ensure_lml_binary()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    edfs = find_chbmit_edfs()
    if args.limit > 0:
        edfs = edfs[: args.limit]
    print(f"[bench_chbmit] {len(edfs)} EDFs, "
          f"{sum(p.stat().st_size for p in edfs) / 1024**3:.2f} GB total",
          file=sys.stderr)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    results: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="bench_chbmit_") as scratch_str:
        scratch = Path(scratch_str)
        for mode in modes:
            print(f"[bench_chbmit] ── starting mode={mode} ──", file=sys.stderr)
            results.append(bench_mode(edfs, mode, scratch))

    # Pick the best CR for the paper headline.
    best = max(results, key=lambda r: r["cr"])
    summary = {
        "dataset": "CHB-MIT Scalp EEG Database",
        "edfs_count": len(edfs),
        "results": results,
        "best_mode": best["mode"],
        "best_cr": best["cr"],
    }

    out_json = OUT_DIR / "chbmit_lpc_mode_compare.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n")
    print()
    print(f"[bench_chbmit] wrote {out_json}")
    print(f"  ── CR summary across modes ──")
    for r in results:
        print(f"    {r['mode']:>8s}  CR={r['cr']:.4f}  "
              f"in={r['input_bytes']:,}  out={r['output_bytes']:,}  "
              f"thrpt={r['throughput_mibs']:.0f} MiB/s")
    print(f"  ── winner: {best['mode']} (CR={best['cr']:.4f}) ──")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
