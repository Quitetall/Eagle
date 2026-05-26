#!/usr/bin/env python3
"""Empirical Shannon entropy on TUH EEG corpora.

Computes two quantities per corpus used by the paper §II.D
("Shannon Entropy Bound for Clinical EEG"):

  H_raw   per-sample entropy of the raw 16-bit signed sample
          stream, aggregated over every channel of every sampled
          EDF as a single global histogram.

  H_diff  per-sample entropy of the first-order-difference
          residual x[t] - x[t-1] (i.e. LPC order-1 with a fixed
          coefficient of 1) — the cheapest possible decorrelating
          predictor, useful as an entropy bound on what any
          first-order predictive coder can achieve.

  CR_raw_ceiling   = 16 / H_raw    (16-bit input)
  CR_diff_ceiling  = 16 / H_diff   (Shannon bound after LPC-1)

These are *upper bounds on the compression ratio* that any
lossless coder with a perfect entropy coder can achieve on the
sampled stream. The measured CR of LamQuant Lossless (~2.28:1 on
TUEG) sits well below the CR_diff ceiling because (a) LamQuant
runs 4-band lifting + variable-order LPC, leaving residuals far
better-decorrelated than first-order, and (b) Golomb-Rice has a
small constant overhead relative to a perfect arithmetic coder.

Output:
  outputs/paper/shannon_entropy_<corpus>.json   per corpus
  outputs/paper/shannon_entropy_summary.json    aggregate table

Sampling: by default, picks --files-per-corpus EDFs uniformly at
random from each --tree; defaults to 200 files which covers a
broad cross-section without taking hours. Each file contributes
*every sample of every channel* to the histogram.

Usage:
  python3 tools/bench_shannon_entropy.py \
    --tree tuar:/mnt/4tb/data/Archive/edf/tuh_repair/tuar_v3.0.1 \
    --tree tueg:/mnt/4tb/data/Archive/edf/tuh_repair/tueg_v2.0.1 \
    --files-per-corpus 200

  # or use --auto-tuh to enumerate all 7 TUH corpora under
  # /mnt/4tb/data/Archive/edf/tuh_repair/ automatically.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "outputs" / "paper"
TUH_REPAIR = Path("/mnt/4tb/data/Archive/edf/tuh_repair")
RNG_SEED = 0x4C414D51  # 'LAMQ' — deterministic sampling

# pyedflib is the EDF reader used elsewhere in the project; it
# returns digital (raw int16) samples via read_digital_signal.
try:
    import pyedflib  # type: ignore
except ImportError as exc:  # pragma: no cover
    sys.stderr.write(f"[shannon] pyedflib import failed: {exc}\n")
    raise


def entropy_from_counts(counts: dict[int, int] | Counter) -> float:
    """Shannon entropy in bits/sample from a histogram."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    h = 0.0
    inv = 1.0 / total
    for c in counts.values():
        if c > 0:
            p = c * inv
            h -= p * math.log2(p)
    return h


def accumulate_file(
    edf_path: Path,
    raw_hist: Counter,
    diff_hist: Counter,
) -> tuple[int, int]:
    """Read all channels of `edf_path`, update both histograms.

    Returns (n_samples_raw_added, n_samples_diff_added).
    """
    n_raw = 0
    n_diff = 0
    with pyedflib.EdfReader(str(edf_path)) as r:
        n_ch = r.signals_in_file
        for ch in range(n_ch):
            # read_digital_signal returns raw int (digital) samples
            x = r.readSignal(ch, digital=True).astype(np.int32, copy=False)
            # Raw histogram: bincount-friendly via collections.Counter on
            # int values. np.unique returns sorted unique values + counts.
            vals, counts = np.unique(x, return_counts=True)
            for v, c in zip(vals.tolist(), counts.tolist()):
                raw_hist[int(v)] += int(c)
            n_raw += int(x.size)

            if x.size >= 2:
                d = np.diff(x)
                vals_d, counts_d = np.unique(d, return_counts=True)
                for v, c in zip(vals_d.tolist(), counts_d.tolist()):
                    diff_hist[int(v)] += int(c)
                n_diff += int(d.size)
    return n_raw, n_diff


def bench_corpus(
    name: str,
    tree: Path,
    files_per_corpus: int,
    log: callable,
) -> dict:
    edfs = sorted(
        p for p in tree.rglob("*.edf") if ".seizures" not in p.name
    )
    if not edfs:
        log(f"[{name}] no EDFs under {tree}")
        return {
            "corpus": name, "tree": str(tree), "files_sampled": 0,
            "n_samples_raw": 0, "n_samples_diff": 0,
            "H_raw": 0.0, "H_diff": 0.0,
            "cr_raw_ceiling": 0.0, "cr_diff_ceiling": 0.0,
        }

    rng = random.Random(RNG_SEED)
    if files_per_corpus > 0 and len(edfs) > files_per_corpus:
        edfs = rng.sample(edfs, files_per_corpus)
    log(f"[{name}] sampling {len(edfs)} EDFs from {tree}")

    raw_hist: Counter = Counter()
    diff_hist: Counter = Counter()
    n_raw_total = 0
    n_diff_total = 0
    failures = 0
    t0 = time.time()
    for i, edf in enumerate(edfs):
        try:
            n_raw, n_diff = accumulate_file(edf, raw_hist, diff_hist)
            n_raw_total += n_raw
            n_diff_total += n_diff
        except Exception as exc:
            failures += 1
            log(f"[{name}] FAIL {edf.name}: {exc}")
        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            log(f"[{name}] {i + 1}/{len(edfs)} ({elapsed:.0f}s, "
                f"fail={failures}, raw_bins={len(raw_hist)})")

    h_raw = entropy_from_counts(raw_hist)
    h_diff = entropy_from_counts(diff_hist)
    cr_raw_ceiling = 16.0 / h_raw if h_raw > 0 else 0.0
    cr_diff_ceiling = 16.0 / h_diff if h_diff > 0 else 0.0
    elapsed = time.time() - t0
    log(f"[{name}] done in {elapsed:.0f}s: "
        f"H_raw={h_raw:.3f} bits, H_diff={h_diff:.3f} bits, "
        f"CR_raw≤{cr_raw_ceiling:.3f}:1, CR_diff≤{cr_diff_ceiling:.3f}:1")

    return {
        "corpus": name,
        "tree": str(tree),
        "files_sampled": len(edfs) - failures,
        "files_failed": failures,
        "n_samples_raw": n_raw_total,
        "n_samples_diff": n_diff_total,
        "raw_alphabet_size": len(raw_hist),
        "diff_alphabet_size": len(diff_hist),
        "H_raw": h_raw,
        "H_diff": h_diff,
        "cr_raw_ceiling": cr_raw_ceiling,
        "cr_diff_ceiling": cr_diff_ceiling,
        "wall_seconds": elapsed,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tree", action="append", default=[],
        help="name:path pair; e.g. tuar:/mnt/4tb/.../tuar_v3.0.1. Repeat.",
    )
    ap.add_argument(
        "--auto-tuh", action="store_true",
        help="auto-enumerate all corpora under /mnt/4tb/data/Archive/"
             "edf/tuh_repair/<corpus>_vX.Y.Z",
    )
    ap.add_argument(
        "--files-per-corpus", type=int, default=200,
        help="sample N EDFs per corpus (0 = all). Default 200.",
    )
    args = ap.parse_args()

    tasks: list[tuple[str, Path]] = []
    for spec in args.tree:
        if ":" not in spec:
            print(f"[shannon] bad --tree spec: {spec}", file=sys.stderr)
            return 1
        name, p = spec.split(":", 1)
        tasks.append((name, Path(p)))

    if args.auto_tuh:
        for child in sorted(TUH_REPAIR.iterdir()):
            if child.is_dir() and "_v" in child.name:
                # name = strip _vX.Y.Z suffix
                base = child.name.split("_v", 1)[0]
                tasks.append((base, child))

    if not tasks:
        print("[shannon] no --tree or --auto-tuh given", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    summary = {"per_corpus": {}, "aggregate": None}
    for name, tree in tasks:
        per = bench_corpus(name, tree, args.files_per_corpus, log)
        summary["per_corpus"][name] = per
        out = OUT_DIR / f"shannon_entropy_{name}.json"
        out.write_text(json.dumps(per, indent=2) + "\n")
        log(f"[{name}] wrote {out}")

    # Cross-corpus aggregate: sum sample counts + entropy weighted by samples
    n_raw = sum(p["n_samples_raw"] for p in summary["per_corpus"].values())
    n_diff = sum(p["n_samples_diff"] for p in summary["per_corpus"].values())
    h_raw_agg = (
        sum(p["H_raw"] * p["n_samples_raw"]
            for p in summary["per_corpus"].values()) / max(n_raw, 1)
    )
    h_diff_agg = (
        sum(p["H_diff"] * p["n_samples_diff"]
            for p in summary["per_corpus"].values()) / max(n_diff, 1)
    )
    summary["aggregate"] = {
        "n_samples_raw": n_raw,
        "n_samples_diff": n_diff,
        "H_raw_weighted": h_raw_agg,
        "H_diff_weighted": h_diff_agg,
        "cr_raw_ceiling": 16.0 / h_raw_agg if h_raw_agg > 0 else 0.0,
        "cr_diff_ceiling": 16.0 / h_diff_agg if h_diff_agg > 0 else 0.0,
    }
    (OUT_DIR / "shannon_entropy_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print()
    print("─── Per-corpus Shannon entropy + CR ceilings ───")
    print(f"  {'corpus':<10s}  {'files':>6s}  {'H_raw':>7s}  "
          f"{'H_diff':>7s}  {'CR_raw':>8s}  {'CR_diff':>8s}")
    for name, p in summary["per_corpus"].items():
        print(f"  {name:<10s}  {p['files_sampled']:>6d}  "
              f"{p['H_raw']:>7.3f}  {p['H_diff']:>7.3f}  "
              f"{p['cr_raw_ceiling']:>7.3f}:1  "
              f"{p['cr_diff_ceiling']:>7.3f}:1")
    a = summary["aggregate"]
    print(f"  {'AGGREGATE':<10s}  {'':>6s}  "
          f"{a['H_raw_weighted']:>7.3f}  {a['H_diff_weighted']:>7.3f}  "
          f"{a['cr_raw_ceiling']:>7.3f}:1  "
          f"{a['cr_diff_ceiling']:>7.3f}:1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
