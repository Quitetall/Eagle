#!/usr/bin/env python3
"""Per-subset compression bench on TUEG v2.0.1.

For the LamQuant TBioCAS paper Appendix~A: produces per-corpus +
per-montage compression numbers (input bytes, compressed bytes,
CR) so the per-subset Table~A.I can be populated from real data.

TUEG ships under `edf/<subj>/...` and subjects fan out by reference
year + session + montage (e.g. `01_tcp_ar`, `02_tcp_le`,
`03_tcp_ar_a`, `04_tcp_le_a`). This script aggregates by montage
suffix because that's the dimension most relevant for codec CR
(channel count + referencing scheme).

For a paper Appendix-A-friendly summary that also reports
per-corpus numbers, point `--tree` at the parent
`/mnt/4tb/data/Archive/edf/tuh_repair/` and aggregate across
each `<corpus>_v*` subtree.

Output:
    outputs/paper/tueg_subset_breakdown.json
    + stdout summary table.

Usage:
    python3 tools/bench_tueg_subsets.py \
        --tree /mnt/4tb/data/Archive/edf/tuh_repair/tueg_v2.0.1

    # Per-corpus mode (any tuh_repair subtree):
    python3 tools/bench_tueg_subsets.py \
        --tree /mnt/4tb/data/Archive/edf/tuh_repair \
        --group-by corpus
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LML_BIN = REPO_ROOT / "target" / "release" / "lml"
OUT_DIR = REPO_ROOT / "outputs" / "paper"

MONTAGE_RE = re.compile(r"(0[1-9]_tcp_(?:ar|le)(?:_a)?)")
CORPUS_RE = re.compile(r"^(tu[a-z]+)_v\d+\.\d+\.\d+$")


def ensure_lml_binary() -> None:
    if LML_BIN.is_file():
        return
    print("[bench_tueg] building lml binary…", file=sys.stderr)
    subprocess.check_call(
        ["cargo", "build", "--release", "--bin", "lml",
         "--manifest-path", str(REPO_ROOT / "Cargo.toml")],
        cwd=REPO_ROOT,
    )


def group_key(edf: Path, tree: Path, mode: str) -> str:
    """Map an EDF path to a subset key."""
    rel = edf.relative_to(tree).as_posix()
    if mode == "montage":
        m = MONTAGE_RE.search(rel)
        return m.group(1) if m else "other"
    if mode == "corpus":
        # First path component should be `<corpus>_vX.Y.Z`.
        head = rel.split("/", 1)[0]
        return head if CORPUS_RE.match(head) else "other"
    raise SystemExit(f"unknown group-by mode: {mode}")


def encode_one(edf: Path, scratch: Path) -> int:
    out_path = scratch / (edf.stem + ".lml")
    subprocess.check_call(
        [str(LML_BIN), "encode", str(edf), "-o", str(out_path),
         "--bare-lml", "--i-understand-data-loss"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    size = out_path.stat().st_size
    out_path.unlink()
    return size


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", type=Path, required=True,
                    help="Root directory of the EDF tree to bench.")
    ap.add_argument("--group-by", choices=("montage", "corpus"),
                    default="montage", help="Aggregation dimension.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Stop after N files (smoke test).")
    args = ap.parse_args()

    if not args.tree.is_dir():
        print(f"[bench_tueg] missing tree: {args.tree}", file=sys.stderr)
        return 1
    ensure_lml_binary()

    edfs = sorted(p for p in args.tree.rglob("*.edf")
                  if ".seizures" not in p.name)
    if args.limit > 0:
        edfs = edfs[: args.limit]
    print(f"[bench_tueg] {len(edfs)} EDFs under {args.tree}",
          file=sys.stderr)

    per_group: dict[str, dict[str, int]] = defaultdict(
        lambda: {"files": 0, "input_bytes": 0, "output_bytes": 0}
    )
    failures = 0
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="bench_tueg_") as scratch_str:
        scratch = Path(scratch_str)
        for i, edf in enumerate(edfs):
            key = group_key(edf, args.tree, args.group_by)
            try:
                in_sz = edf.stat().st_size
                out_sz = encode_one(edf, scratch)
                bucket = per_group[key]
                bucket["files"] += 1
                bucket["input_bytes"] += in_sz
                bucket["output_bytes"] += out_sz
            except subprocess.CalledProcessError as e:
                failures += 1
                print(f"[bench_tueg] FAIL {edf.name}: {e}", file=sys.stderr)
            if (i + 1) % 200 == 0:
                el = time.time() - t0
                print(f"[bench_tueg] {i+1}/{len(edfs)} ({el:.0f}s, fail={failures})",
                      file=sys.stderr)

    summary = {
        "tree": str(args.tree),
        "group_by": args.group_by,
        "files_total": sum(g["files"] for g in per_group.values()),
        "files_failed": failures,
        "groups": {},
    }
    for key, g in sorted(per_group.items()):
        cr = g["input_bytes"] / max(g["output_bytes"], 1)
        summary["groups"][key] = {**g, "cr": cr}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"tueg_subset_breakdown_{args.group_by}.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")

    print()
    print(f"[bench_tueg] wrote {out}")
    print(f"  ── per-{args.group_by} breakdown ──")
    print(f"  {'group':<24s}  {'files':>7s}  {'in (GB)':>10s}  "
          f"{'out (GB)':>10s}  {'CR':>6s}")
    for key, g in sorted(summary["groups"].items()):
        in_gb = g["input_bytes"] / 1024**3
        out_gb = g["output_bytes"] / 1024**3
        print(f"  {key:<24s}  {g['files']:>7d}  {in_gb:>10.2f}  "
              f"{out_gb:>10.2f}  {g['cr']:>5.3f}:1")

    total_in = sum(g["input_bytes"] for g in summary["groups"].values())
    total_out = sum(g["output_bytes"] for g in summary["groups"].values())
    print(f"  {'TOTAL':<24s}  {summary['files_total']:>7d}  "
          f"{total_in/1024**3:>10.2f}  {total_out/1024**3:>10.2f}  "
          f"{total_in/max(total_out,1):>5.3f}:1")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
