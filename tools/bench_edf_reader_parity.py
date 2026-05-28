#!/usr/bin/env python3
"""Cross-EDF-reader parity validation for the LamQuant Lossless codec.

For the LamQuant TBioCAS paper (Phase 2, item 12): documents the
forensic-parity verification protocol claimed in §IV.B. Two
independent EDF reader implementations are used to confirm that the
codec's reconstructed EDFs are bit-exact AND semantically faithful
(channel labels, sample rate, sample data) to the originals:

1. pyedflib (C-backed; EDF/EDF+ spec reference reader)
2. mne.io.read_raw_edf (Python; MNE-Python standard reader)

Methodology per sample:
* sha256(original.edf) == sha256(roundtrip.edf)
* pyedflib: channel count, channel labels, sample rates, total
  samples, per-channel digital min/max all match.
* MNE: signal arrays match exactly after both readers normalize.

Output:
    outputs/paper/edf_reader_parity.json — full audit with per-file
    pass/fail + summary counts.

Usage:
    python3 tools/bench_edf_reader_parity.py \
        --tree /mnt/4tb/data/Archive/edf/tuh_repair/tueg_v2.0.1 \
        --pyedflib-samples 500 --mne-samples 4000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys

if sys.version_info < (3, 10):
    sys.exit(f"bench_edf_reader_parity.py requires Python 3.10+, detected {sys.version_info.major}.{sys.version_info.minor}")
import tempfile
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
LML_BIN = REPO_ROOT / "target" / "release" / "lml"
OUT_DIR = REPO_ROOT / "outputs" / "paper"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def roundtrip(edf_in: Path, scratch: Path) -> Path:
    """Encode + decode `edf_in` through the codec. Return the
    decoded EDF path. Raises on failure."""
    lml_out = scratch / (edf_in.stem + ".lml")
    edf_out = scratch / (edf_in.stem + ".rt.edf")
    subprocess.check_call(
        [str(LML_BIN), "encode", str(edf_in), "-o", str(lml_out),
         "--bare-lml", "--i-understand-data-loss"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    subprocess.check_call(
        [str(LML_BIN), "decode", str(lml_out), "-o", str(edf_out),
         "--to-edf"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    lml_out.unlink()
    return edf_out


def check_pyedflib(orig: Path, recon: Path) -> dict:
    """Compare via pyedflib: structural metadata + per-channel
    digital samples must match exactly."""
    import pyedflib  # type: ignore[import-not-found]
    issues: list[str] = []
    with pyedflib.EdfReader(str(orig)) as a, pyedflib.EdfReader(str(recon)) as b:
        if a.signals_in_file != b.signals_in_file:
            issues.append(f"channel count {a.signals_in_file} vs {b.signals_in_file}")
        if a.getSignalLabels() != b.getSignalLabels():
            issues.append("channel labels differ")
        if list(a.getSampleFrequencies()) != list(b.getSampleFrequencies()):
            issues.append("sample rates differ")
        if not issues:
            for ch in range(a.signals_in_file):
                xa = a.readSignal(ch, digital=True)
                xb = b.readSignal(ch, digital=True)
                if not np.array_equal(xa, xb):
                    issues.append(f"ch{ch} digital samples diff")
                    break
    return {"reader": "pyedflib", "pass": len(issues) == 0, "issues": issues}


def check_mne(orig: Path, recon: Path) -> dict:
    """Compare via MNE-Python: physical signal arrays must match
    within numerical equality."""
    import mne  # type: ignore[import-not-found]
    import logging
    logging.getLogger("mne").setLevel(logging.ERROR)
    issues: list[str] = []
    a = mne.io.read_raw_edf(str(orig), preload=True, verbose=False)
    b = mne.io.read_raw_edf(str(recon), preload=True, verbose=False)
    if a.info["nchan"] != b.info["nchan"]:
        issues.append(f"nchan {a.info['nchan']} vs {b.info['nchan']}")
    elif not np.allclose(a.get_data(), b.get_data(), rtol=0, atol=0):
        issues.append("signal arrays differ")
    return {"reader": "mne", "pass": len(issues) == 0, "issues": issues}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", type=Path, required=True,
                    help="Directory tree of original EDFs to sample.")
    ap.add_argument("--pyedflib-samples", type=int, default=500)
    ap.add_argument("--mne-samples", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20260526)
    args = ap.parse_args()

    if not LML_BIN.exists():
        print("[bench_edf_parity] building lml binary…", file=sys.stderr)
        subprocess.check_call(
            ["cargo", "build", "--release", "--bin", "lml",
             "--manifest-path", str(REPO_ROOT / "Cargo.toml")],
            cwd=REPO_ROOT,
        )

    all_edfs = sorted(p for p in args.tree.rglob("*.edf")
                      if ".seizures" not in p.name)
    if not all_edfs:
        print(f"[bench_edf_parity] no .edf under {args.tree}", file=sys.stderr)
        return 1
    print(f"[bench_edf_parity] {len(all_edfs)} candidate EDFs", file=sys.stderr)

    rng = random.Random(args.seed)
    py_sample = rng.sample(all_edfs, min(args.pyedflib_samples, len(all_edfs)))
    mne_sample = rng.sample(all_edfs, min(args.mne_samples, len(all_edfs)))

    pyedflib_results: list[dict] = []
    mne_results: list[dict] = []
    failures: list[dict] = []

    with tempfile.TemporaryDirectory(prefix="bench_edf_parity_") as scratch_str:
        scratch = Path(scratch_str)
        t0 = time.time()

        # pyedflib pass
        for i, edf in enumerate(py_sample):
            try:
                recon = roundtrip(edf, scratch)
                if sha256(edf) != sha256(recon):
                    failures.append({"file": str(edf), "stage": "sha256",
                                     "msg": "byte-mismatch"})
                    continue
                res = check_pyedflib(edf, recon)
                pyedflib_results.append(res)
                if not res["pass"]:
                    failures.append({"file": str(edf), "stage": "pyedflib",
                                     "issues": res["issues"]})
                recon.unlink()
            except Exception as e:
                failures.append({"file": str(edf), "stage": "pyedflib-exc",
                                 "msg": str(e)})
            if (i + 1) % 50 == 0:
                el = time.time() - t0
                print(f"[bench_edf_parity] pyedflib {i+1}/{len(py_sample)} "
                      f"({el:.0f}s, fails={len(failures)})", file=sys.stderr)

        # MNE pass (independent sample)
        t1 = time.time()
        for i, edf in enumerate(mne_sample):
            try:
                recon = roundtrip(edf, scratch)
                if sha256(edf) != sha256(recon):
                    failures.append({"file": str(edf), "stage": "sha256",
                                     "msg": "byte-mismatch"})
                    continue
                res = check_mne(edf, recon)
                mne_results.append(res)
                if not res["pass"]:
                    failures.append({"file": str(edf), "stage": "mne",
                                     "issues": res["issues"]})
                recon.unlink()
            except Exception as e:
                failures.append({"file": str(edf), "stage": "mne-exc",
                                 "msg": str(e)})
            if (i + 1) % 100 == 0:
                el = time.time() - t1
                print(f"[bench_edf_parity] mne {i+1}/{len(mne_sample)} "
                      f"({el:.0f}s, fails={len(failures)})", file=sys.stderr)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "edf_reader_parity.json"
    audit = {
        "tree": str(args.tree),
        "candidate_files": len(all_edfs),
        "pyedflib": {
            "samples_attempted": len(py_sample),
            "samples_passed": sum(1 for r in pyedflib_results if r["pass"]),
        },
        "mne": {
            "samples_attempted": len(mne_sample),
            "samples_passed": sum(1 for r in mne_results if r["pass"]),
        },
        "failures": failures,
        "seed": args.seed,
    }
    out.write_text(json.dumps(audit, indent=2) + "\n")
    print()
    print(f"[bench_edf_parity] wrote {out}")
    print(f"  pyedflib pass: {audit['pyedflib']['samples_passed']}/{audit['pyedflib']['samples_attempted']}")
    print(f"  mne pass:      {audit['mne']['samples_passed']}/{audit['mne']['samples_attempted']}")
    print(f"  failures:      {len(failures)}")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
