#!/usr/bin/env python3
"""Codec-agnostic BCI motor-imagery downstream-task concordance bench.

The BCI parallel to Eagle's NEURAL seizure-detection concordance
(`tests/validation/test_downstream_concordance.py`, whose implementation
lives in the sibling LamQuant-Neural tree). Where the seizure bench asks
"does the codec preserve seizure-detection F1/AUROC?", this bench asks the
motor-imagery question:

    Does a compression codec preserve downstream BCI *decoding* accuracy?

It answers it the only honest way — end to end:

    X (trials) ──quantize──▶ int16 ──┬─ write→read EDF ───────────────▶ Xq  (quantized reference)
                                     └─ codec roundtrip (encode/extract/
                                        decode) ─────────────────────▶ Xr  (codec output)

Both branches are decoded with the SAME CSP+LDA pipeline + the same CV
seed, and the accuracy/kappa deltas are compared. For a LOSSLESS codec the
deltas are EXACTLY 0.0 because the round-trip is bit-exact (Xq == Xr); the
bench is the load-bearing scaffold that earns its keep when the
neural / lossy codec ships and the deltas can go nonzero.

Design constraints (mirrors the NEURAL concordance + the Rust adapter):
  * The CORE harness depends only on numpy + mne + sklearn + pyedflib +
    the `lml` BINARY. It is codec-agnostic — the codec is an opaque
    compress/decompress box reached through `codec_roundtrip_edf` (or any
    injected `roundtrip_fn`).
  * MOABB is OPTIONAL. It is imported LAZILY inside `load_moabb` and is
    NEVER imported at module top-level, so the module — and the offline
    tests — import and run with no moabb, no codec wheel, and no network.
  * The lossless Δ=0 claim measures CODEC distortion, not quantization:
    the reference branch Xq is the EDF-loaded original (post-int16
    quantization), so the delta isolates exactly what the codec changed.

Public API (all importable + unit-testable, no moabb at top level):
    decoding_score(X, y, sfreq, *, seed=0) -> dict
    resolve_lml_bin() -> str | None
    quantize_to_int16(X) -> np.ndarray (int16)
    codec_roundtrip_edf(X_int, sfreq, ch_names, lml_bin) -> np.ndarray
    concordance(X, y, sfreq, ch_names, *, lml_bin, seed=0, roundtrip_fn=None) -> dict
    synthetic_bci_dataset(...) -> (X, y, sfreq, ch_names)
    load_moabb(dataset=..., paradigm=..., subject=...) -> (X, y, sfreq, ch_names)
    main(argv=None) -> int

Usage:
    # offline, codec-transparent on a real lml binary:
    LML_BIN=/path/to/lml python3 tools/bench_moabb_concordance.py --source synthetic

    # real motor-imagery data (needs the `moabb` extra + the `data` marker):
    pip install -e '.[moabb]'
    python3 tools/bench_moabb_concordance.py --source moabb \
        --dataset BNCI2014_001 --paradigm LeftRightImagery --subject 1
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

if sys.version_info < (3, 10):
    sys.exit(
        "bench_moabb_concordance.py requires Python 3.10+, detected "
        f"{sys.version_info.major}.{sys.version_info.minor}"
    )

import numpy as np

# NOTE: moabb / pyriemann are NOT imported here. moabb is reached only
# through load_moabb() (lazy). mne + sklearn + pyedflib are hard deps of
# the core harness and may be imported at top level, but to keep the
# module importable in the thinnest possible env we defer mne/sklearn into
# the functions that need them too (so `import bench_moabb_concordance`
# never pulls the decoding stack unless decoding is actually run).

REPO_ROOT = Path(__file__).resolve().parents[1]

# int16 digital range. Writing trials to EDF with
# physical_min/max == digital_min/max == [INT16_MIN, INT16_MAX] makes the
# digital→physical transform the identity, so int16 sample values pass
# through the EDF UNSCALED (phys == digital). This is what makes the
# lossless round-trip bit-exact at the *sample* level.
INT16_MIN = -32768
INT16_MAX = 32767
# Headroom: scale the float peak to 32000 (not 32767) so the rint+clip
# never saturates an edge sample into a different code.
INT16_SCALE_PEAK = 32000.0

# Acceptance: |accuracy_delta| < this ⇒ codec is "downstream transparent".
# A lossless codec hits exactly 0.0; the threshold matters for lossy codecs.
TRANSPARENT_ACCURACY_DELTA = 0.01

# Datasets/paradigms known to work with load_moabb's get_data() path.
# (dataset_class_name, paradigm_class_name) — both live in moabb.
SUPPORTED_MOABB = {
    "BNCI2014_001": ("LeftRightImagery", "MotorImagery"),
    "BNCI2014_004": ("LeftRightImagery", "MotorImagery"),
    "Zhou2016": ("LeftRightImagery", "MotorImagery"),
    "PhysionetMI": ("LeftRightImagery", "MotorImagery"),
}


# ---------------------------------------------------------------------------
# lml binary resolution — mirrors the Rust adapter + bench_edf_reader_parity.
# ---------------------------------------------------------------------------
def resolve_lml_bin() -> str | None:
    """Resolve the `lml` codec binary, in the same order the Rust adapter uses.

    1. ``$LML_BIN`` (if it points at an existing file)
    2. sibling ``../LamQuant-Lossless/target/{release,debug}/lml``
    3. ``shutil.which("lml")`` (anything on ``$PATH``)

    Returns the absolute path as a str, or None if no binary is found.
    """
    env = os.environ.get("LML_BIN")
    if env and Path(env).is_file():
        return str(Path(env).resolve())

    sib = REPO_ROOT.parent / "LamQuant-Lossless" / "target"
    for profile in ("release", "debug"):
        cand = sib / profile / "lml"
        if cand.is_file():
            return str(cand.resolve())

    on_path = shutil.which("lml")
    if on_path:
        return on_path
    return None


# ---------------------------------------------------------------------------
# Decoding: CSP + LDA, cross-validated. mne + sklearn imported here (not top).
# ---------------------------------------------------------------------------
def decoding_score(X, y, sfreq, *, seed: int = 0) -> dict:
    """Cross-validated CSP→LDA motor-imagery decoding score.

    Pipeline: ``mne.decoding.CSP(n_components=4, reg='ledoit_wolf')`` →
    ``sklearn LinearDiscriminantAnalysis``, evaluated with
    ``StratifiedKFold(5, shuffle=True, random_state=seed)``.

    Args:
        X: float array ``[n_trials, n_channels, n_times]``.
        y: integer class labels ``[n_trials]`` (2 classes).
        sfreq: sampling frequency (Hz). Accepted for API symmetry; CSP
            on covariance features does not use it.
        seed: CV shuffle seed (also fixes the fold assignment so
            ``accuracy`` and ``kappa`` are computed over the same folds).

    Returns:
        ``{"accuracy": float, "kappa": float}`` where ``accuracy`` is the
        mean of ``cross_val_score`` and ``kappa`` is
        ``cohen_kappa_score`` on the out-of-fold ``cross_val_predict``.
    """
    from mne.decoding import CSP
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import cohen_kappa_score
    from sklearn.model_selection import (
        StratifiedKFold,
        cross_val_predict,
        cross_val_score,
    )
    from sklearn.pipeline import Pipeline

    logging.getLogger("mne").setLevel(logging.ERROR)

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)

    csp = CSP(n_components=4, reg="ledoit_wolf", log=True)
    clf = Pipeline([("csp", csp), ("lda", LinearDiscriminantAnalysis())])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    accuracy = float(np.mean(cross_val_score(clf, X, y, cv=cv)))
    y_pred = cross_val_predict(clf, X, y, cv=cv)
    kappa = float(cohen_kappa_score(y, y_pred))
    return {"accuracy": accuracy, "kappa": kappa}


# ---------------------------------------------------------------------------
# Deterministic int16 quantization (per-recording fixed scale).
# ---------------------------------------------------------------------------
def quantize_to_int16(X) -> np.ndarray:
    """Scale/clip float ``X`` into the int16 range, deterministically.

    A single per-recording scale (driven by the global peak amplitude)
    maps the float trials into ``[-32768, 32767]``. The SAME quantization
    is applied to both branches of the concordance (the EDF reference and
    the codec round-trip), so the delta measures the CODEC, not the
    quantization. Pure function of ``X`` — no RNG, no global state.

    Args:
        X: float array of any shape.

    Returns:
        int16 array, same shape as ``X``.
    """
    X = np.asarray(X, dtype=np.float64)
    peak = float(np.max(np.abs(X))) if X.size else 0.0
    if peak == 0.0:
        peak = 1.0
    scale = INT16_SCALE_PEAK / peak
    Xi = np.rint(X * scale)
    Xi = np.clip(Xi, INT16_MIN, INT16_MAX)
    return Xi.astype(np.int16)


# ---------------------------------------------------------------------------
# EDF I/O helpers (pyedflib). int16 pass-through (phys == digital).
# ---------------------------------------------------------------------------
def _trials_to_edf(X_int: np.ndarray, sfreq: float, ch_names, edf_path: Path) -> None:
    """Write trials concatenated along time into ONE EDF (int16, unscaled).

    physical_min/max == digital_min/max == [INT16_MIN, INT16_MAX] so the
    digital→physical map is the identity and int16 codes survive verbatim.
    Trials are laid out channel-major as ``[n_ch, n_trials * n_times]``.
    """
    import pyedflib

    n_trials, n_ch, n_times = X_int.shape
    # [n_trials, n_ch, n_times] -> [n_ch, n_trials*n_times]
    flat = np.ascontiguousarray(X_int.transpose(1, 0, 2).reshape(n_ch, n_trials * n_times))

    writer = pyedflib.EdfWriter(str(edf_path), n_ch, file_type=pyedflib.FILETYPE_EDFPLUS)
    try:
        headers = [
            {
                "label": str(ch_names[i])[:16] or f"EEG{i}",
                "dimension": "uV",
                "sample_frequency": float(sfreq),
                "physical_min": float(INT16_MIN),
                "physical_max": float(INT16_MAX),
                "digital_min": INT16_MIN,
                "digital_max": INT16_MAX,
                "transducer": "",
                "prefilter": "",
            }
            for i in range(n_ch)
        ]
        writer.setSignalHeaders(headers)
        # digital=True requires integer arrays; int32 is what pyedflib wants.
        writer.writeSamples([flat[i].astype(np.int32) for i in range(n_ch)], digital=True)
    finally:
        writer.close()


def _edf_to_trials(edf_path: Path, n_trials: int, n_ch: int, n_times: int) -> np.ndarray:
    """Read an EDF back into ``[n_trials, n_ch, n_times]`` int16 digital values."""
    import pyedflib

    reader = pyedflib.EdfReader(str(edf_path))
    try:
        rows = [reader.readSignal(i, digital=True) for i in range(n_ch)]
    finally:
        reader.close()
    flat = np.rint(np.vstack(rows)).astype(np.int16)  # [n_ch, n_trials*n_times]
    return flat.reshape(n_ch, n_trials, n_times).transpose(1, 0, 2)


def _read_edf_reference(X_int: np.ndarray, sfreq: float, ch_names) -> np.ndarray:
    """Write→read X_int through one EDF (NO codec). The quantized reference Xq.

    This is the branch the codec is compared against: identical EDF
    write/read path, minus the codec, so any nonzero delta is attributable
    to the codec and not to EDF/pyedflib quirks.
    """
    n_trials, n_ch, n_times = X_int.shape
    with tempfile.TemporaryDirectory(prefix="moabb_concordance_ref_") as scratch:
        edf = Path(scratch) / "ref.edf"
        _trials_to_edf(X_int, sfreq, ch_names, edf)
        return _edf_to_trials(edf, n_trials, n_ch, n_times)


# ---------------------------------------------------------------------------
# Codec round-trip via the lml binary. encode → .lma → extract → decode.
# ---------------------------------------------------------------------------
def codec_roundtrip_edf(X_int: np.ndarray, sfreq: float, ch_names, lml_bin: str) -> np.ndarray:
    """Round-trip int16 trials through the codec and read them back.

    Pipeline (each step verified bit-exact upstream by ``lml roundtrip``):
        1. write trials to ONE EDF (int16, unscaled — see ``_trials_to_edf``)
        2. ``lml encode in.edf -o out.lma``     (per-recording .lma archive)
        3. ``lml extract out.lma -o ex/``       (recovers the .lml signal)
        4. ``lml decode <.lml> -o rt.edf --to-edf``  (byte-exact EDF)
        5. read rt.edf back via pyedflib (digital=True)

    Deterministic: no RNG, fixed temp layout, fixed flags. Raises
    ``subprocess.CalledProcessError`` on any nonzero lml exit and
    ``RuntimeError`` if an expected output file is missing.

    Args:
        X_int: int16 array ``[n_trials, n_channels, n_times]``.
        sfreq: sampling frequency (Hz).
        ch_names: channel labels.
        lml_bin: path to the lml binary (see ``resolve_lml_bin``).

    Returns:
        int16 array ``[n_trials, n_channels, n_times]`` — the codec output.
    """
    X_int = np.asarray(X_int)
    if X_int.dtype != np.int16:
        raise TypeError(f"codec_roundtrip_edf expects int16, got {X_int.dtype}")
    n_trials, n_ch, n_times = X_int.shape

    def _run(cmd):
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(
                proc.returncode, cmd, stderr=proc.stderr.decode("utf-8", "replace")
            )

    with tempfile.TemporaryDirectory(prefix="moabb_concordance_codec_") as scratch:
        d = Path(scratch)
        edf_in = d / "trials.edf"
        lma = d / "trials.lma"
        ex_dir = d / "extracted"
        edf_rt = d / "trials.rt.edf"

        _trials_to_edf(X_int, sfreq, ch_names, edf_in)

        _run([lml_bin, "encode", str(edf_in), "-o", str(lma)])
        if not lma.is_file():
            raise RuntimeError(f"lml encode produced no archive at {lma}")

        _run([lml_bin, "extract", str(lma), "-o", str(ex_dir)])
        lmls = sorted(ex_dir.rglob("*.lml"))
        if not lmls:
            raise RuntimeError(f"lml extract produced no .lml under {ex_dir}")

        _run([lml_bin, "decode", str(lmls[0]), "-o", str(edf_rt), "--to-edf"])
        if not edf_rt.is_file():
            raise RuntimeError(f"lml decode --to-edf produced no EDF at {edf_rt}")

        return _edf_to_trials(edf_rt, n_trials, n_ch, n_times)


# ---------------------------------------------------------------------------
# The headline: end-to-end downstream concordance.
# ---------------------------------------------------------------------------
def concordance(X, y, sfreq, ch_names, *, lml_bin, seed: int = 0, roundtrip_fn=None) -> dict:
    """Compress→decompress→decode→compare. The downstream-task concordance.

    Builds two decoded scores from the SAME quantized int16 trials:
      * ``Xq`` — the quantized reference: ``X`` → int16 → write/read EDF
        (no codec). This is what the codec output is compared against, so
        the delta isolates CODEC distortion (not quantization).
      * ``Xr`` — the codec round-trip output (``roundtrip_fn`` if given,
        else the real ``codec_roundtrip_edf`` driving the lml binary).

    Both are scored with ``decoding_score`` at the SAME ``seed`` (so the
    folds line up and the delta is purely a function of the samples).

    For a LOSSLESS codec ``Xq == Xr`` bit-for-bit ⇒ every delta is exactly
    0.0 and ``bit_exact`` is True.

    Args:
        X: float trials ``[n_trials, n_channels, n_times]``.
        y: class labels.
        sfreq: sampling frequency (Hz).
        ch_names: channel labels.
        lml_bin: path to lml binary (used only when ``roundtrip_fn`` is None).
        seed: CV seed, shared by both branches.
        roundtrip_fn: optional injectable codec. Signature
            ``fn(X_int, sfreq, ch_names) -> int16 array``. Lets a fake
            lossy codec be wired in for testing the metric is not
            trivially always-zero.

    Returns:
        dict with accuracy_orig/recon/delta, kappa_orig/recon/delta,
        bit_exact (bool), transparent (bool), n_trials, n_channels.
    """
    X = np.asarray(X, dtype=np.float64)
    n_trials, n_channels, _ = X.shape

    X_int = quantize_to_int16(X)

    # Reference branch: same EDF write/read, no codec.
    Xq = _read_edf_reference(X_int, sfreq, ch_names)

    # Codec branch.
    if roundtrip_fn is None:
        if not lml_bin:
            raise ValueError("concordance: lml_bin required when roundtrip_fn is None")
        Xr = codec_roundtrip_edf(X_int, sfreq, ch_names, lml_bin)
    else:
        Xr = np.asarray(roundtrip_fn(X_int, sfreq, ch_names))

    if Xr.shape != Xq.shape:
        raise RuntimeError(f"round-trip shape {Xr.shape} != reference shape {Xq.shape}")

    score_orig = decoding_score(Xq, y, sfreq, seed=seed)
    score_recon = decoding_score(Xr, y, sfreq, seed=seed)

    accuracy_delta = score_recon["accuracy"] - score_orig["accuracy"]
    kappa_delta = score_recon["kappa"] - score_orig["kappa"]

    return {
        "accuracy_orig": score_orig["accuracy"],
        "accuracy_recon": score_recon["accuracy"],
        "accuracy_delta": accuracy_delta,
        "kappa_orig": score_orig["kappa"],
        "kappa_recon": score_recon["kappa"],
        "kappa_delta": kappa_delta,
        "bit_exact": bool(np.array_equal(Xq, Xr)),
        "transparent": bool(abs(accuracy_delta) < TRANSPARENT_ACCURACY_DELTA),
        "n_trials": int(n_trials),
        "n_channels": int(n_channels),
    }


# ---------------------------------------------------------------------------
# Synthetic, genuinely CSP-decodable, motor-imagery-shaped dataset.
# ---------------------------------------------------------------------------
def synthetic_bci_dataset(
    n_trials: int = 40,
    n_ch: int = 8,
    n_times: int = 256,
    sfreq: float = 128.0,
    seed: int = 0,
):
    """Two-class, CSP-decodable synthetic motor-imagery dataset (pure numpy).

    Each class injects a band-limited (~12 Hz mu-band) oscillation into a
    DISTINCT, fixed spatial mixing vector on top of broadband background
    noise. Because the discriminative power lives in the spatial
    covariance structure — exactly what CSP whitens and separates — clean
    decoding accuracy is clearly above chance (~1.0 in practice). No moabb,
    no network, fully reproducible from ``seed``.

    Returns:
        ``(X, y, sfreq, ch_names)`` with ``X`` float64
        ``[n_trials, n_ch, n_times]``, ``y`` int labels (balanced 0/1),
        ``ch_names`` ``["EEG0"..."EEG{n_ch-1}"]``.
    """
    if n_trials % 2 != 0:
        raise ValueError("n_trials must be even (balanced two-class)")
    rng = np.random.default_rng(seed)
    y = np.array([0, 1] * (n_trials // 2), dtype=int)
    t = np.arange(n_times) / sfreq

    # Two distinct unit spatial topographies — the class signature.
    A0 = rng.standard_normal(n_ch)
    A0 /= np.linalg.norm(A0)
    A1 = rng.standard_normal(n_ch)
    A1 /= np.linalg.norm(A1)

    freq = 12.0  # mu-band, motor imagery ERD/ERS-ish
    X = np.zeros((n_trials, n_ch, n_times), dtype=np.float64)
    for i in range(n_trials):
        background = rng.standard_normal((n_ch, n_times)) * 0.5
        phase = rng.uniform(0.0, 2.0 * np.pi)
        osc = np.sin(2.0 * np.pi * freq * t + phase)
        amp = rng.uniform(2.5, 3.5)
        spatial = A0 if y[i] == 0 else A1
        X[i] = background + amp * np.outer(spatial, osc)

    ch_names = [f"EEG{i}" for i in range(n_ch)]
    return X, y, float(sfreq), ch_names


# ---------------------------------------------------------------------------
# Real motor-imagery data — moabb imported LAZILY, only here.
# ---------------------------------------------------------------------------
def load_moabb(
    dataset: str = "BNCI2014_001",
    paradigm: str = "LeftRightImagery",
    subject: int = 1,
):
    """Load a real motor-imagery dataset via moabb (imported lazily, here).

    moabb is OPTIONAL. It is imported inside this function body — importing
    this module never requires moabb. If moabb is absent, a clear
    ``ImportError`` with an install hint is raised (never a bare traceback).

    Supported ``(dataset, paradigm)`` pairs (see ``SUPPORTED_MOABB``):
        BNCI2014_001 / LeftRightImagery | MotorImagery
        BNCI2014_004 / LeftRightImagery | MotorImagery
        Zhou2016     / LeftRightImagery | MotorImagery
        PhysionetMI  / LeftRightImagery | MotorImagery

    Args:
        dataset: moabb dataset class name.
        paradigm: moabb paradigm class name.
        subject: subject id (1-indexed, dataset-specific).

    Returns:
        ``(X, y, sfreq, ch_names)`` — float trials, int labels, sfreq, labels.

    Raises:
        ImportError: moabb not installed (with a ``pip install`` hint).
        ValueError: unsupported (dataset, paradigm) pair.
    """
    try:
        import moabb  # noqa: F401  (lazy: never imported at module top level)
        import moabb.datasets as moabb_datasets
        import moabb.paradigms as moabb_paradigms
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise ImportError(
            "load_moabb requires the optional 'moabb' extra, which is not "
            "installed. Install it with:\n"
            "    pip install -e '.[moabb]'\n"
            "  (or: pip install 'moabb>=1.0' 'pyriemann>=0.5')\n"
            "The synthetic + lossless offline path does NOT need moabb."
        ) from exc

    if dataset not in SUPPORTED_MOABB:
        raise ValueError(
            f"unsupported moabb dataset {dataset!r}; "
            f"supported: {sorted(SUPPORTED_MOABB)}"
        )
    if paradigm not in SUPPORTED_MOABB[dataset]:
        raise ValueError(
            f"unsupported paradigm {paradigm!r} for dataset {dataset!r}; "
            f"supported: {list(SUPPORTED_MOABB[dataset])}"
        )

    dataset_obj = getattr(moabb_datasets, dataset)()
    paradigm_obj = getattr(moabb_paradigms, paradigm)()

    X, labels, _meta = paradigm_obj.get_data(dataset=dataset_obj, subjects=[subject])

    # Map string labels → contiguous int codes, deterministically.
    classes = sorted(set(labels))
    code = {c: i for i, c in enumerate(classes)}
    y = np.array([code[v] for v in labels], dtype=int)

    # The authoritative sfreq is the dataset's; paradigms resample to it.
    sfreq = float(getattr(paradigm_obj, "resample", None) or _moabb_sfreq(dataset_obj))
    ch_names = list(getattr(paradigm_obj, "channels", None) or _moabb_channels(X))

    return np.asarray(X, dtype=np.float64), y, sfreq, ch_names


def _moabb_sfreq(dataset_obj) -> float:
    """Best-effort sfreq from a moabb dataset object. Falls back to 250 Hz."""
    for attr in ("sfreq", "fs"):
        val = getattr(dataset_obj, attr, None)
        if val:
            return float(val)
    return 250.0


def _moabb_channels(X) -> list:
    """Generate channel labels when the paradigm doesn't expose them."""
    n_ch = np.asarray(X).shape[1]
    return [f"EEG{i}" for i in range(n_ch)]


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def _print_report(result: dict, source: str) -> None:
    bar = "=" * 62
    print(bar)
    print("BCI DECODING CONCORDANCE")
    print(bar)
    print(f"  source           : {source}")
    print(f"  trials           : {result['n_trials']}")
    print(f"  channels         : {result['n_channels']}")
    print("  ----------------------------------------------------------")
    print(f"  accuracy_orig    : {result['accuracy_orig']:.6f}")
    print(f"  accuracy_recon   : {result['accuracy_recon']:.6f}")
    print(f"  accuracy_delta   : {result['accuracy_delta']:+.6f}")
    print(f"  kappa_orig       : {result['kappa_orig']:.6f}")
    print(f"  kappa_recon      : {result['kappa_recon']:.6f}")
    print(f"  kappa_delta      : {result['kappa_delta']:+.6f}")
    print(f"  bit_exact        : {result['bit_exact']}")
    print(f"  transparent (<{TRANSPARENT_ACCURACY_DELTA}) : {result['transparent']}")
    print(bar)
    if result["bit_exact"]:
        print("  VERDICT: LOSSLESS — codec is downstream-transparent (Δ == 0).")
    elif result["transparent"]:
        print("  VERDICT: TRANSPARENT — |Δaccuracy| within tolerance.")
    else:
        print("  VERDICT: DEGRADED — codec changes downstream decoding.")
    print(bar)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Codec-agnostic BCI motor-imagery downstream concordance bench."
    )
    ap.add_argument("--source", choices=["synthetic", "moabb"], default="synthetic")
    ap.add_argument("--dataset", default="BNCI2014_001")
    ap.add_argument("--paradigm", default="LeftRightImagery")
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--lml-bin", default=None, help="Override the lml binary path.")
    ap.add_argument("--out", type=Path, default=None, help="Write the result dict as JSON here.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    lml_bin = args.lml_bin or resolve_lml_bin()
    if not lml_bin:
        print(
            "[bci_concordance] no lml binary found. Set $LML_BIN, clone "
            "LamQuant-Lossless as a sibling and build it, or put `lml` on "
            "$PATH.",
            file=sys.stderr,
        )
        return 2

    if args.source == "synthetic":
        X, y, sfreq, ch_names = synthetic_bci_dataset(seed=args.seed)
    else:
        try:
            X, y, sfreq, ch_names = load_moabb(
                dataset=args.dataset, paradigm=args.paradigm, subject=args.subject
            )
        except ImportError as exc:
            # No traceback — a clear, actionable message + nonzero exit.
            print(f"[bci_concordance] {exc}", file=sys.stderr)
            return 3
        except (ValueError, Exception) as exc:  # noqa: BLE001 - harness boundary
            print(f"[bci_concordance] moabb load failed: {exc}", file=sys.stderr)
            return 4

    try:
        result = concordance(X, y, sfreq, ch_names, lml_bin=lml_bin, seed=args.seed)
    except Exception as exc:  # noqa: BLE001 - harness boundary, surface cleanly
        print(f"[bci_concordance] concordance failed: {exc}", file=sys.stderr)
        return 5

    _print_report(result, args.source)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"[bci_concordance] wrote {args.out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
