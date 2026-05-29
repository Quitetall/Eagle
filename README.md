# Eagle

LamQuant validation + benchmarking suite. Reproduces every numerical claim in the *IEEE Transactions on Biomedical Circuits and Systems* paper "LamQuant Lossless: A Real-Time, Bit-Exact, Wirelessly-Deployable EEG Compression Algorithm" (2026 submission).

## What's in here

| Subdir | Purpose |
|---|---|
| `tools/bench_chbmit.py` | CHB-MIT subset compression-ratio + latency bench (Table II / IV in paper) |
| `tools/bench_tueg_subsets.py` | Per-corpus TUEG breakdown (Appendix A) |
| `tools/bench_per_file_cr.py` | Per-file CR distribution + boxplot data (Figure 3) |
| `tools/bench_shannon_entropy.py` | Empirical entropy ceiling H_0, H_0(ΔX) for §II.D |
| `tools/bench_edf_reader_parity.py` | MNE/pyedflib cross-validation (§IV.B four-layer protocol) |
| `tools/bench_gzip_baseline.sh` | gzip/zstd baseline comparison on TUEG sample |
| `tools/verify_paper_claims.py` | Single-shot script that walks every paper number → confirms against bundled `evidence/*.json` (60 PASS / 0 FAIL on a freshly-bundled supplementary) |
| `tools/hazard3_bench/` | Rust no_std bench harness for RP2350 Hazard3 — produces `bench_encode.elf` for Verilator RTL simulation (paper §IV.D measured-cycle row) |
| `tools/bench/bench_rs/` | Host-side criterion benches (codec hot paths) |
| `tests/benchmarks/` | Compression-ratio + ablation + biological-fidelity + C-parity tests |
| `tests/validation/` | EDF cross-check, NEDC seizure-detection downstream concordance |
| `tests/audits/` | Format consistency audits across export paths |

## Dependencies

Eagle depends on **[LamQuant-Lossless](https://github.com/Quitetall/LamQuant-Lossless)** (the codec under test). It is a **standalone** clone; no sibling layout required. The Rust bench crates pull `lamquant-firmware` directly over a pinned git-rev, so Eagle resolves on its own.

```bash
git clone git@github.com:Quitetall/Eagle.git
cd Eagle

# Python tools — install the codec wheel from LamQuant-Lossless
pip install "lamquant-codec @ git+https://github.com/Quitetall/LamQuant-Lossless.git#subdirectory=reference_implementations/python_codec"
pip install -e .

# Rust hazard3_bench (riscv32 firmware target)
cd tools/hazard3_bench && cargo build --target riscv32imac-unknown-none-elf

# Rust host benches
cd ../bench/bench_rs && cargo build
```

Both `tools/hazard3_bench/Cargo.toml` and `tools/bench/bench_rs/Cargo.toml` reference `lamquant-firmware` via a pinned git-rev:

```toml
lamquant-firmware = { git = "https://github.com/Quitetall/LamQuant-Lossless.git", rev = "abcae4794c38b3d3e75d3c214063cf0307e3daba", default-features = false }
```

Pulling a specific rev keeps the bench numbers reproducible against an exact codec commit. Bump the `rev` to re-pin against a newer LamQuant-Lossless release.

## How to reproduce paper numbers

```bash
# 1. Bench a corpus (CHB-MIT shown; substitute path as needed)
python3 tools/bench_chbmit.py --corpus /path/to/chbmit --out evidence/

# 2. (Optional) bench TUEG subsets — requires 1.76 TB local mirror
python3 tools/bench_tueg_subsets.py --corpus /path/to/tueg --out evidence/

# 3. (Optional) Verilator RTL bench for RP2350 cycle count
cd tools/hazard3_bench && bash run_bench.sh verilator

# 4. Verify every paper claim against evidence JSON
python3 tools/verify_paper_claims.py
# Expected: 60 PASS / 0 FAIL when evidence/ is fully populated
```

## Two modes — external LQS vs internal LamQuant dev

Eagle's test/bench suite runs in two clearly-namespaced modes:

**External — the LamQuant Standard (LQS).** Default. Codec-agnostic: the
vendor-neutral **LQS** standard (its own repo, [Quitetall/LQS](https://github.com/Quitetall/LQS),
consumed here by the `eagle` crate) plus the agnostic Python suites treat any
conforming codec as an opaque compress/decompress box and verify only
externally-observable quantities (compression ratio, bit-exact round-trip,
latency, EDF parity). These run without the neural/torch stack and are what
external CI executes:

```bash
pytest -m "not internal"     # external LQS Python suite
cargo test -p eagle          # the eagle crate (pulls the sibling LQS standard)
```

**Internal — LamQuant vendor dev.** LamQuant-specific introspection
benchmarks that reach inside the neural codec internals (FSQ entropy/activity,
latent utilization, Cayley rotation, residual-FSQ, subband leakage, TNN memory,
XNOR/cpop MACs, C-vs-Python parity, ablation matrix). They are a dev dependency
of LamQuant, **not** part of the external standard; they require the sibling
**LamQuant-Neural** source tree plus the **LamQuant-Lossless** wheel, stay in
Python, and are gated behind the `internal` pytest marker so they are skipped
by default. See [`tests/internal/README.md`](tests/internal/README.md).

```bash
pip install -e '.[neural]'   # sibling neural + lossless stack
pytest -m internal           # internal LamQuant dev suite
```

The introspection benches physically remain under `tests/benchmarks/`; they are
tagged, not moved.

## Architecture

Eagle is one repo in an 8-product Unix decomposition of LamQuant. It depends on **LamQuant-Lossless** (codec library) and optionally on **LamQuant-Neural** (when the neural codec ships) via Cargo feature flags.

## Rust fast gate (eagle crate + sibling LQS)

The vendor-neutral **LQS** standard is its own crate ([Quitetall/LQS](https://github.com/Quitetall/LQS),
Rust-canonical for fast CI / pre-commit feedback). Eagle's `eagle` crate
consumes it via a sibling-clone path dep — clone LQS next to this repo:

```
parent/
  LQS/      (github.com/Quitetall/LQS)
  Eagle/    (this repo)
```

CI runs the `eagle-rust` job (sibling-checkout LQS, then `cargo build -p eagle`
+ `cargo test -p eagle`). To run the same gate locally before every push,
enable the bundled hook:

```bash
chmod +x scripts/lqs-fastgate.sh .githooks/pre-push
git config core.hooksPath .githooks   # enables .githooks/pre-push (Rust fast gate)
```

`scripts/lqs-fastgate.sh` (`cargo build -q -p eagle && cargo test -q -p eagle`)
is the fast local mirror of the CI `eagle-rust` job; run it standalone anytime.

## License

GNU GENERAL PUBLIC LICENSE v3 (see `LICENSE.md`).
