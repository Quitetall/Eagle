# LQS — The LamQuant Quality Standard

**A vendor-neutral, codec-agnostic benchmark for EEG compression.**

LQS is a specification *and* a runnable reference implementation for grading
EEG codecs. It exists so that any lab can take any codec — lossless, lossy,
neural, classical — wrap it behind one small interface, and obtain a single,
reproducible compliance verdict that is directly comparable to every other
codec graded the same way.

This document is the authoritative spec. Where a number appears here it is
taken verbatim from the implementation it describes; the source of truth for
the thresholds is [`src/levels.rs`](src/levels.rs), for the metrics
[`src/metrics.rs`](src/metrics.rs), for the per-band split
[`src/bands.rs`](src/bands.rs), and for the dispatch
[`src/harness.rs`](src/harness.rs).

---

## 1. What LQS is

LQS is an **external** benchmark. It tests a codec through its public
encode/decode boundary — it never reaches inside.

- **Codec-agnostic.** A codec is anything that turns a per-channel integer
  signal into an opaque byte blob and back. LQS does not know or care whether
  the blob holds an arithmetic-coded residual, a learned latent, a wavelet
  tree, or raw bytes. The only contract is the [`Codec`](src/adapter.rs)
  trait (Section 4).
- **Black-box.** Grading is a function of *observable behaviour only*:
  - the compressed size the codec produced,
  - the signal it reconstructs after a full encode → decode round trip,
  - the wall-clock time and working-set bytes that round trip cost.

  No internal entropy estimates, no claimed bit budgets, no
  "theoretically lossless" assertions are trusted. A codec earns its grade by
  what comes out the other end, measured against the original.

- **Functionality *and* performance.** A run measures fidelity (PRD, PRDN,
  Pearson R, SNR, per-band) *and* resource cost (compression ratio,
  encode+decode throughput in MiB/s, peak working bytes), then bundles the lot
  into one self-describing [`LqsReport`](src/report.rs) that serializes to
  stable JSON — the wire format two labs exchange to compare codecs.

The headline output is a single tier grade: **L**, **C**, **M**, **A**, or
*below-floor*.

---

## 2. The five tiers

A codec is graded against an ordered ladder of tiers. Each tier sets floors
and ceilings on the global metrics, and (for the lossy tiers) per-band
fidelity floors. The grade is the **highest tier the codec fully satisfies**
(Section 5).

| Tier | Name | Global gate | Per-band reqs | Meaning |
|------|------|-------------|---------------|---------|
| **L** | Lossless | PRD **exactly 0** on the integer sample domain **AND** CR ≥ **0.8** | none | Bit-exact reconstruction. The only honest lossless claim. |
| **C** | Clinical | R ≥ **0.95**, PRD ≤ **9.0 %**, CR ≥ **20.0** | δ θ α β γ (below) | Distortion small enough for clinical read. |
| **M** | Monitoring | R ≥ **0.85**, PRD ≤ **20.0 %**, CR ≥ **100.0** | δ θ α β γ (below) | Usable for continuous monitoring. |
| **A** | Alerting | R ≥ **0.70**, PRD ≤ **40.0 %**, CR ≥ **200.0** | δ θ α β γ (below) | Coarse, but events still detectable. |
| **—** | below floor | fails even the A gate | — | Non-compliant. CLI exits non-zero. |

`max_snr_loss` is carried per tier in the table (L: 0.0, C: 3.0, M: 6.0,
A: 10.0 dB) for documentation and reporting parity with the original spec,
but it is **not gated** in this implementation — `snr_db` is measured and
reported, never used to pass or fail a tier.

### 2.1 The L tier — exact zero, not "small PRD"

The Lossless tier is special-cased. It does **not** test a float PRD against
an epsilon. It tests **integer-domain exact equality**: the reconstructed
samples must equal the originals element-for-element
([`metrics::prd_is_exact_zero`](src/metrics.rs)). Exact integer equality *is*
a PRD of exactly zero, with no float round-off.

This distinction is load-bearing. Two integer streams that differ by a single
LSB have a float PRD on the order of 1e-12 — small enough that a naive
`float_prd < eps` gate would wave them through as "lossless". The integer gate
rejects them. A codec that *claims* lossless but is off by one LSB on one
sample is graded as lossy, exactly as it should be.

The L tier additionally requires **CR ≥ 0.8**: a codec that *expands* the data
cannot claim lossless compliance even if its reconstruction is perfect. This
0.8 no-expansion floor is locked by the vendor-neutral spec (see
[`L_TIER_MIN_CR`](src/harness.rs) and the `min_cr` field of the L level in
[`src/levels.rs`](src/levels.rs)).

The L tier has **no per-band requirements**: a bit-exact reconstruction has
zero error in every band by construction, so the per-band table for an L
report is filled with the perfect-by-construction row (R = 1, PRD = 0,
SNR = 120 dB) without paying for a DFT pass.

### 2.2 Per-band fidelity requirements (C / M / A)

Each lossy tier carries a per-band requirement table keyed by band name.
A band requirement constrains a measured band only when a band of that name
is present in the measurement; **unmeasured bands are not penalized**.
The values below are verbatim from [`src/levels.rs`](src/levels.rs).
The `freq_range` column is documentary (it is *not* what the gate consults —
the gate uses only `max_prd` and `min_r`); the actual band split is performed
by [`src/bands.rs`](src/bands.rs) (Section 3.4).

**C — Clinical**

| band | freq_range (Hz) | max PRD | min R |
|------|-----------------|---------|-------|
| delta | 0.5 – 4.0 | 5.0 % | 0.98 |
| theta | 4.0 – 8.0 | 7.0 % | 0.97 |
| alpha | 8.0 – 13.0 | 8.0 % | 0.96 |
| beta | 13.0 – 30.0 | 12.0 % | 0.93 |
| gamma | 30.0 – 50.0 | 20.0 % | 0.85 |

**M — Monitoring**

| band | freq_range (Hz) | max PRD | min R |
|------|-----------------|---------|-------|
| delta | 0.5 – 4.0 | 10.0 % | 0.95 |
| theta | 4.0 – 8.0 | 12.0 % | 0.93 |
| alpha | 8.0 – 13.0 | 15.0 % | 0.90 |
| beta | 13.0 – 30.0 | 25.0 % | 0.80 |
| gamma | 30.0 – 50.0 | 40.0 % | 0.60 |

**A — Alerting**

| band | freq_range (Hz) | max PRD | min R |
|------|-----------------|---------|-------|
| delta | 0.5 – 4.0 | 20.0 % | 0.85 |
| theta | 4.0 – 8.0 | 25.0 % | 0.80 |
| alpha | 8.0 – 13.0 | 30.0 % | 0.75 |
| beta | 13.0 – 30.0 | 40.0 % | 0.65 |
| gamma | 30.0 – 50.0 | 60.0 % | 0.40 |

A band fails its tier if its measured R is below `min_r` **or** its measured
PRD exceeds `max_prd`. A single band failure drops the codec to the next
tier (Section 5).

---

## 3. Metrics

All metric formulas have exactly **one** definition, in
[`src/metrics.rs`](src/metrics.rs), ported from the training-pipeline source
of truth (`ai_models/metrics.py`) and the Python codec mirror
(`lamquant_codec/lqs.py`). The two co-equal *primary* fidelity metrics are
**R** (shape preservation) and **PRD** (magnitude preservation).

Aggregate global metrics are computed over the **flattened** multichannel
signal: channels are concatenated in order into one contiguous vector, then a
single metric is taken over the whole stream
([`flatten_f64`](src/harness.rs)). On a length mismatch every metric truncates
to the shorter slice.

### 3.1 PRD — Percentage Root-mean-square Difference

```
PRD = 100 · sqrt( Σ (x − x̂)²  /  Σ x² )
```

Lower is better. All-zero guard: if `Σ x²` is below 1e-12 the result is `0.0`
when the residual is also ~0, else `100.0`.

For the **L tier only**, PRD is replaced by the integer-domain exact-zero test
`prd_is_exact_zero(orig_i64, recon_i64)` — element-wise integer equality,
returning `false` immediately on any length mismatch — *not* the float PRD
above. (See Section 2.1 for why.)

### 3.2 PRDN — Normalized PRD

```
PRDN = 100 · sqrt( Σ (x − x̂)²  /  Σ (x − mean(x))² )
```

Mean-subtracted denominator, so a large DC offset cannot deflate the reported
error. Same all-zero guard as PRD, applied to the mean-subtracted energy.
Measured and reported; not directly gated.

### 3.3 Pearson R, SNR, CR, QS, Entropy

- **Pearson R** — correlation of the two flattened streams, mean-centred. If
  either input's standard deviation is below 1e-8 the correlation is
  undefined; the function returns `1.0` when the streams are element-wise
  (approximately) equal and `0.0` otherwise. Higher is better.

- **SNR (dB)** — `10 · log10( mean(x²) / mean((x − x̂)²) )`. When the noise
  power is below 1e-30 (effectively perfect reconstruction) the result is
  **capped at 120.0 dB**. Reported, not gated.

- **CR — compression ratio** — `raw_bytes / max(comp_bytes, 1)`. The
  denominator is the codec's blob length; the numerator is the size of the
  signal under the reference container (Section 4.2), so the ratio is
  comparable across codecs and the `Store` baseline reports CR ≈ 1.0 by
  construction. Across a corpus, CR is **pooled by bytes**
  (`Σ raw / Σ comp`, via [`aggregate_cr`](src/metrics.rs)) — the correct way
  to combine ratios; a plain mean would over-weight tiny files.

- **QS — quality score** — `CR / PRD`, higher is better (more compression per
  unit distortion). Guarded: when PRD ≤ 0 (lossless / degenerate) QS is the
  raw CR, so a perfect codec ranks by compression ratio rather than blowing up
  to infinity.

- **Shannon entropy (bits)** — `H = −Σ pᵢ log₂ pᵢ` from a histogram of symbol
  counts ([`entropy_from_counts`](src/metrics.rs)). Zero-count bins
  contribute nothing; an empty or all-zero histogram has entropy 0.0. Provided
  as a utility metric for codec analysis.

### 3.4 Per-band fidelity — same filter, both signals

Per-band fidelity is measured by [`bands::per_band_fidelity`](src/bands.rs).
The method:

1. Take one real-input DFT of the original and one of the reconstruction.
2. For each clinical band, build a **rectangular frequency-domain mask** that
   keeps only the DFT bins whose centre frequency falls in the half-open band
   `[lo, hi)`, and inverse-transform back to the time domain.
3. Measure R, PRD, and SNR on the masked band signals using the same
   one-definition formulas from Section 3.1–3.3.

The **load-bearing property**: the identical mask is applied to *both* the
original and the reconstruction. Because both signals are filtered the same
way, the measured per-band PRD / R / SNR reflect **codec error inside that
band**, never a mismatch between two differently-filtered signals.

The DFT is a hand-rolled O(N²) transform with no FFT dependency (the crate
stays std-only); for the short per-window EEG segments LQS grades this is fast
enough. The masking partitions the spectrum so band reconstructions sum back
to the original (no bin dropped or double-counted).

The clinical band edges actually used by the band split are pinned in
[`CLINICAL_BANDS`](src/bands.rs), inclusive-low / exclusive-high in Hz:

| band | range (Hz) |
|------|------------|
| sub-delta | 0 – 1 |
| delta | 1 – 4 |
| theta | 4 – 8 |
| alpha | 8 – 12 |
| beta | 13 – 30 |
| gamma | 30 – 100 |

> **Note on band edges.** The measurement table (`CLINICAL_BANDS`, six bands
> including `sub-delta`) and the tier-requirement table (the five named bands
> in `src/levels.rs`, whose documentary `freq_range` uses the 0.5–4 / 8–13
> conventions) are two different tables. The grading gate pairs a measured
> band with its tier requirement **by name** — so `delta`, `theta`, `alpha`,
> `beta`, `gamma` are both measured and gated, while `sub-delta` is measured
> and reported but has no tier requirement to gate against. The names are
> deliberately kept identical between the two tables so the pairing works.
> `alpha` uses the 8–12 Hz convention in the measurement split; change that
> row if your protocol pins 8–13.

For a bit-exact (L-tier) report the per-band table is the perfect row
(R = 1, PRD = 0, SNR = 120) for each band, emitted without running the DFT.

---

## 4. The codec adapter protocol — make YOUR codec benchmarkable

To benchmark a codec under LQS you implement one trait. That is the entire
contract.

### 4.1 The `Codec` trait

From [`src/adapter.rs`](src/adapter.rs):

```rust
pub trait Codec {
    /// Short, stable identifier used in reports (e.g. "store").
    fn name(&self) -> &str;

    /// Whether the codec *claims* bit-exact reconstruction.
    fn declared_lossless(&self) -> bool;

    /// Compress a per-channel integer signal into an opaque blob.
    fn encode(&self, signal: &[Vec<i64>], fs: f64) -> Vec<u8>;

    /// Reconstruct the per-channel integer signal from a blob.
    fn decode(&self, blob: &[u8]) -> Vec<Vec<i64>>;
}
```

- `signal` is one `Vec<i64>` of samples per channel (the integer ADC-count
  domain). `fs` is the sample rate in Hz — metadata the codec may use or
  ignore. The reference adapters reconstruct the samples without needing `fs`
  in their byte stream; a real codec is free to embed `fs` (or anything else)
  in its own blob.
- `encode` returns an opaque `Vec<u8>`; `decode` inverts it. LQS imposes no
  structure on the blob.

### 4.2 Verify, don't trust — the lossless claim is graded, not assumed

`declared_lossless()` is a **claim**, not a fact the harness accepts. The
harness verifies the claim against the L-tier gate by actually comparing the
reconstruction to the original on the integer domain. A codec that returns
`declared_lossless() == true` but reconstructs a single sample off by one LSB
is reported `bit_exact = false` and graded as lossy. This "verify the claim,
do not trust it" principle is the integrity guarantee of the whole standard.

A decode that returns garbage (wrong length, wrong values, or an empty `Vec`
on a malformed blob) simply surfaces to the L-tier gate as a length/value
mismatch and fails the lossless claim — which is the correct outcome.

### 4.3 Reference serialization (used by the bundled adapters)

The shipped reference adapters share one tiny, deterministic, byte-exact
container ([`serialize`] / [`deserialize`] in `src/adapter.rs`). You are not
required to use it — it exists so the reference adapters have a common
denominator and so two backends can be checked for byte-equality. Layout (all
integers little-endian):

```text
magic   : 4 bytes  = b"LQS0"
n_chan  : u32       = number of channels
per chan: u32 len, then `len` × i64 samples
```

`fs` is deliberately **not** in this stream — it is metadata the harness
already holds.

---

## 5. Grade dispatch

Grading happens in [`harness::run`](src/harness.rs), which drives one
encode + decode round trip, measures the resource cost, then dispatches:

1. **Lossless gate (short-circuit).** Flatten the original and the
   reconstruction to integer streams and test
   `prd_is_exact_zero(orig_i64, recon_i64)`. If it is **bit-exact AND
   CR ≥ 0.8**, the codec earns grade **L** and the lossy battery is *skipped
   entirely* — there is no distortion to measure. PRD = 0, R = 1, SNR = 120,
   per-band perfect-by-construction.

2. **Lossy battery — C → M → A descent.** Otherwise convert to `f64`, compute
   the global metrics (PRD, PRDN, R, SNR) over the flattened signal, build the
   per-band table, and hand the lot to [`levels::grade`](src/levels.rs). A
   lossy tier passes **iff** the global R, PRD, and CR thresholds are all met
   **AND** every *measured* band meets its per-band R and PRD floors. The gate
   tries the tiers in strictness order **C, then M, then A**, and returns the
   **first (highest) tier that fully passes**.

3. **Below floor.** If no lossy tier passes, the grade is the below-floor
   sentinel (`'\0'`, rendered as `—` / empty string). The codec is
   non-compliant.

**Strictness order: L < C < M < A < below-floor** (L is strongest;
[`grade_rank`](src/harness.rs)).

**The climb-a-tier to-do list.** When a codec settles on, say, M, the report's
`violations` field carries the reasons the *strictest tier it attempted* (C)
failed — a precise to-do list for climbing a tier (e.g.
`"global PRD 15.00% > 9.00%"`, `"gamma R 0.7000 < 0.8500"`). The list is empty
when the codec already passes the strictest tier.

**Corpus runs.** [`harness::run_corpus`](src/harness.rs) grades a corpus of
files for one codec, returns one report per file plus a `CorpusSummary`. The
summary pools CR by bytes, averages PRD and R across files, and reports the
**worst** (lowest) grade observed — a codec's corpus compliance is bounded by
its weakest file.

**Leaderboard.** [`report::leaderboard`](src/report.rs) ranks codecs
best-first: **grade first** (a stronger tier always outranks a weaker one),
then by QS descending, then CR descending, with codec name as the
deterministic final tie-break.

---

## 6. Running it

The crate ships a CLI front-end, `eagle-lqs`:

```bash
cargo run -p lqs --bin eagle-lqs -- <codec>
```

`<codec>` is one of:

- `store` — identity passthrough, the CR ≈ 1.0 lossless baseline (default if
  no argument is given),
- `gzip` — pure-Rust gzip (miniz_oxide via `flate2`), the always-available
  compressed lossless reference,
- `quantize` — a deliberately-lossy demo codec (÷8 on encode, ×8 on decode)
  that lands in the lossy tiers, included to exercise the C/M/A battery.

The CLI runs the harness over a **built-in synthetic multichannel EEG-like
signal** (4 channels × 512 samples @ 256 Hz, a deterministic sum of sinusoids
placed in distinct clinical bands), prints the human-readable report table and
a one-line `LQS-<grade> COMPLIANT` badge, and **exits non-zero if the codec is
below the alerting floor**.

> **EDF input is not yet wired.** Real-EDF / corpus loading from a file is a
> documented `TODO` in [`src/bin/eagle_lqs.rs`](src/bin/eagle_lqs.rs); the CLI
> currently always grades the built-in fixture so it runs anywhere with zero
> external data. The library `harness::run` / `harness::run_corpus` API already
> accepts arbitrary `(Vec<Vec<i64>>, fs)` signals, so an EDF reader can feed it
> directly today — only the CLI argument parsing for a file path is pending.

### Reference adapters

Three lossless reference adapters ship in the library so the suite has
something to grade out of the box:

- **`Store`** — `encode` = `serialize`, `decode` = `deserialize`. No
  compression; the CR ≈ 1.0 baseline every other codec's ratio is measured
  against.
- **`Gzip`** — `serialize` then gzip-compress (pure-Rust, no system zlib).
  Bit-exact lossless.
- **`Zstd`** — optional, compiled in only under the `zstd` Cargo feature
  (level 19), so the default build carries no system dependency. Bit-exact
  lossless.

All three declare lossless and round-trip the channel layout bit-exactly
(including empty channels, ragged channel lengths, and the i64 extremes).

---

## 7. Rust is canonical; Python is a mirror

The **Rust** crate (this directory) is the **canonical** implementation of the
LQS standard. The tier thresholds, the metric formulas, and the dispatch logic
here are authoritative.

A **Python mirror** lives at
`reference_implementations/python_codec/lamquant_codec/lqs.py`. It exposes the
same standard (`LQS_LEVELS`, `run_compliance`, the `BandRequirement` /
`LQSLevel` records) so the Python training/eval pipeline can grade against LQS
without a Rust dependency. The C / M / A tier tables — `max_prd`, `min_r`,
`min_cr`, `max_snr_loss`, and the per-band δ/θ/α/β/γ `max_prd` / `min_r`
values — are kept in **parity** with the Rust table and are ported between the
two verbatim (see the module header of [`src/levels.rs`](src/levels.rs)).

Two intentional differences exist where the Rust implementation tightens the
standard:

- **L-tier PRD gate.** Rust redefines the lossless gate as the
  integer-domain *exact-zero* short-circuit
  ([`prd_is_exact_zero`](src/metrics.rs)); the Python mirror's float-PRD path
  is the looser ancestor. The exact-zero test is the correct lossless contract
  (Section 2.1).
- **L-tier `min_cr`.** Rust locks the lossless no-expansion floor to
  **0.8**; the Python mirror's L level carries a different `min_cr`. The 0.8
  value is the vendor-neutral spec floor and is authoritative.

When the canonical Rust thresholds change, the Python mirror's tier tables
must be updated to match — that parity is the contract between the two
implementations.
