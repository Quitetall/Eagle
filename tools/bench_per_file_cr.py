#!/usr/bin/env python3
"""Per-file CR distribution for the LamQuant Lossless paper.

For each EDF in the given tree, runs `lml encode --bare-lml` to
size-only mode and records the per-file (raw_bytes, comp_bytes,
cr) triple. Outputs:

  outputs/paper/per_file_cr_<corpus>.json
    {
      "corpus": str, "tree": str, "files": int,
      "raw_bytes_total": int, "comp_bytes_total": int,
      "aggregate_cr": float,
      "distribution": {
        "min": float, "p5": float, "p25": float, "median": float,
        "p75": float, "p95": float, "max": float, "mean": float,
        "stdev": float, "n_below_1": int,
        "n_above_5": int, "n_above_10": int
      },
      "outliers_top": [{"path","raw_bytes","comp_bytes","cr"}, ...],
      "outliers_bottom": [...],
      "histogram": [{ "bin_low": float, "bin_high": float, "count": int }, ...]
    }

The paper consumes the distribution + outliers to substantiate
the per-file CR claim in §IV.C (range from near-1:1 on noisy
recordings up to dozens:1 on low-noise segments).

Usage:
  python3 tools/bench_per_file_cr.py \
      --tree tueg-edf000:/mnt/4tb/data/Archive/edf/tuh_repair/\
tueg_v2.0.1/edf/000 \
      --sample 0
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
import sys

if sys.version_info < (3, 10):
    sys.exit(f"bench_per_file_cr.py requires Python 3.10+, detected {sys.version_info.major}.{sys.version_info.minor}")
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LML_BIN = REPO_ROOT / "target" / "release" / "lml"
OUT_DIR = REPO_ROOT / "outputs" / "paper"
RNG_SEED = 0x4C414D51


def ensure_lml() -> None:
    if LML_BIN.is_file():
        return
    print("[per_file_cr] building lml binary…", file=sys.stderr)
    subprocess.check_call(
        ["cargo", "build", "--release", "--bin", "lml",
         "--manifest-path", str(REPO_ROOT / "Cargo.toml")],
        cwd=REPO_ROOT,
    )


def encode_one(edf: Path, scratch: Path) -> int:
    """Encode `edf` to `scratch`, return compressed file size."""
    out_path = scratch / (edf.stem + ".lml")
    subprocess.check_call(
        [str(LML_BIN), "encode", str(edf), "-o", str(out_path),
         "--bare-lml", "--i-understand-data-loss"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    size = out_path.stat().st_size
    out_path.unlink()
    return size


def histogram(crs: list[float], n_bins: int = 20) -> list[dict]:
    """Linear histogram across [0, max_cr]."""
    if not crs:
        return []
    lo = 0.0
    hi = max(crs)
    width = (hi - lo) / n_bins
    bins = [0] * n_bins
    for cr in crs:
        idx = min(int((cr - lo) / width), n_bins - 1) if width > 0 else 0
        bins[idx] += 1
    return [
        {"bin_low": lo + i * width,
         "bin_high": lo + (i + 1) * width,
         "count": bins[i]}
        for i in range(n_bins)
    ]


def bench_corpus(
    name: str,
    tree: Path,
    sample: int,
    log,
) -> dict:
    edfs = sorted(p for p in tree.rglob("*.edf")
                  if ".seizures" not in p.name)
    if not edfs:
        log(f"[{name}] no EDFs under {tree}")
        return {"corpus": name, "tree": str(tree), "files": 0}

    rng = random.Random(RNG_SEED)
    if sample > 0 and len(edfs) > sample:
        edfs = rng.sample(edfs, sample)
    log(f"[{name}] encoding {len(edfs)} EDFs from {tree}")

    records: list[dict] = []
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix=f"per_file_cr_{name}_") as sc:
        scratch = Path(sc)
        for i, edf in enumerate(edfs):
            try:
                raw = edf.stat().st_size
                comp = encode_one(edf, scratch)
                cr = raw / max(comp, 1)
                records.append({
                    "path": str(edf.relative_to(tree)),
                    "raw_bytes": raw,
                    "comp_bytes": comp,
                    "cr": cr,
                })
            except Exception as exc:
                log(f"[{name}] FAIL {edf.name}: {exc}")
            if (i + 1) % 50 == 0:
                el = time.time() - t0
                log(f"[{name}] {i + 1}/{len(edfs)} ({el:.0f}s)")

    crs = [r["cr"] for r in records]
    raw_total = sum(r["raw_bytes"] for r in records)
    comp_total = sum(r["comp_bytes"] for r in records)

    if not crs:
        return {"corpus": name, "tree": str(tree), "files": 0}

    crs_sorted = sorted(crs)
    n = len(crs_sorted)

    def pct(p: float) -> float:
        idx = max(0, min(n - 1, int(round(p * (n - 1) / 100.0))))
        return crs_sorted[idx]

    distribution = {
        "min": crs_sorted[0],
        "p5": pct(5),
        "p25": pct(25),
        "median": pct(50),
        "p75": pct(75),
        "p95": pct(95),
        "max": crs_sorted[-1],
        "mean": sum(crs_sorted) / n,
        "stdev": statistics.stdev(crs_sorted) if n > 1 else 0.0,
        "n_below_1_5": sum(1 for c in crs if c < 1.5),
        "n_below_2": sum(1 for c in crs if c < 2.0),
        "n_above_3": sum(1 for c in crs if c > 3.0),
        "n_above_5": sum(1 for c in crs if c > 5.0),
        "n_above_10": sum(1 for c in crs if c > 10.0),
    }

    records.sort(key=lambda r: r["cr"], reverse=True)
    outliers_top = records[:5]
    outliers_bottom = list(reversed(records[-5:]))

    result = {
        "corpus": name,
        "tree": str(tree),
        "files": n,
        "raw_bytes_total": raw_total,
        "comp_bytes_total": comp_total,
        "aggregate_cr": raw_total / max(comp_total, 1),
        "distribution": distribution,
        "outliers_top": outliers_top,
        "outliers_bottom": outliers_bottom,
        "histogram": histogram(crs, n_bins=20),
    }
    log(f"[{name}] CR range {distribution['min']:.3f}–{distribution['max']:.3f}, "
        f"median {distribution['median']:.3f}, "
        f"aggregate {result['aggregate_cr']:.3f}")
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", action="append", default=[],
                    help="name:path; repeatable.")
    ap.add_argument("--sample", type=int, default=0,
                    help="Sample N files per tree (0 = all).")
    args = ap.parse_args()

    if not args.tree:
        print("[per_file_cr] no --tree given", file=sys.stderr)
        return 1
    ensure_lml()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    for spec in args.tree:
        name, path = spec.split(":", 1)
        result = bench_corpus(name, Path(path), args.sample, log)
        out = OUT_DIR / f"per_file_cr_{name}.json"
        out.write_text(json.dumps(result, indent=2) + "\n")
        log(f"[{name}] wrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
