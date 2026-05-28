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

Eagle depends on **[LamQuant-Lossless](https://github.com/Quitetall/LamQuant-Lossless)** (the codec under test). Two install paths:

### Option A — Sibling-clone (recommended for dev)

```bash
mkdir lamquant && cd lamquant
git clone git@github.com:Quitetall/LamQuant-Lossless.git
git clone git@github.com:Quitetall/Eagle.git

# Python tools
cd LamQuant-Lossless/reference_implementations/python_codec && pip install -e .
cd ../../../Eagle && pip install -r requirements.txt  # (TODO Phase 4.1: ship requirements.txt)

# Rust hazard3_bench (assumes sibling layout)
cd Eagle/tools/hazard3_bench
cargo build --target riscv32imac-unknown-none-elf
```

`tools/hazard3_bench/Cargo.toml` references `lamquant-firmware` via relative path `../../../LamQuant-Lossless/lamquant-firmware` — sibling layout is the contract.

### Option B — Git dep (when LamQuant-Lossless goes public)

`tools/hazard3_bench/Cargo.toml` will switch to:
```toml
lamquant-firmware = { git = "https://github.com/Quitetall/LamQuant-Lossless.git", branch = "main" }
```

This switch happens when LamQuant-Lossless flips public. Until then, Option A is the only path.

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

**External — the LamQuant Standard (LQS).** Default. Codec-agnostic: the Rust
crate under `lqs/` plus the agnostic Python suites treat any conforming codec
as an opaque compress/decompress box and verify only externally-observable
quantities (compression ratio, bit-exact round-trip, latency, EDF parity).
These run without the neural/torch stack and are what external CI executes:

```bash
pytest -m "not internal"     # external LQS Python suite
cargo test -p lqs            # external LQS Rust crate (see lqs/ section below)
```

**Internal — LamQuant vendor dev.** LamQuant-specific introspection
benchmarks that reach inside the neural codec internals (FSQ entropy/activity,
latent utilization, Cayley rotation, residual-FSQ, subband leakage, TNN memory,
XNOR/cpop MACs, C-vs-Python parity, ablation matrix). They are a dev dependency
of LamQuant, **not** part of the external standard — they require the sibling
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

## LQS Rust crate — local fast gate

`lqs/` is the vendor-neutral EEG codec benchmark standard, Rust-canonical for
fast CI / pre-commit feedback. CI runs the `lqs-rust` job (build + test +
`eagle-lqs store` smoke). To run the same gate locally before every push,
enable the bundled hook:

```bash
chmod +x scripts/lqs-fastgate.sh .githooks/pre-push
git config core.hooksPath .githooks   # enables .githooks/pre-push (LQS fast gate)
```

`scripts/lqs-fastgate.sh` (`cd lqs && cargo build -q && cargo test -q`) is the
fast local mirror of the CI `lqs-rust` job; run it standalone anytime.

## License

GNU GENERAL PUBLIC LICENSE v3 (see `LICENSE.md`).
