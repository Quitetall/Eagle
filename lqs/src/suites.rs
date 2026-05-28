//! Codec-agnostic benchmark batteries — the native Rust port of the
//! black-box benchmark *logic* that used to live in Eagle's Python
//! `tests/benchmarks/` + `tools/`.
//!
//! These are the batteries that exercise a codec purely through its
//! [`crate::adapter::Codec`] byte interface — encode, decode, measure.
//! They never reach inside a model: no FSQ symbols, no latents, no Cayley
//! rotations, no rANS state. (Those internal-introspection benches stay in
//! Python, behind `-m internal`.) Because every battery here operates over
//! the same `&dyn Codec` + `&[(signal, fs)]` corpus, the canonical fast
//! path (this crate) now covers them, and any codec — neural, classical,
//! hybrid — runs through the identical measurement.
//!
//! Each battery is a **pure function** of `(&dyn Codec, &[(Vec<Vec<i64>>,
//! f64)])`: no global state, no I/O, no checkpoint discovery. The corpus is
//! one `(per-channel integer signal, sample rate)` tuple per file, matching
//! [`crate::harness::run_corpus`].
//!
//! Ported Python bench logics:
//!
//! | Rust battery                  | Python source(s)                          |
//! |-------------------------------|-------------------------------------------|
//! | [`rate_distortion`]           | `benchmark_rate_distortion.py` (the (rate, R, PRD) point any codec produces) + the per-file `(cr, R, PRD)` rows of `benchmark_compression_ratio.py` |
//! | [`per_file_cr_distribution`]  | `tools/bench_per_file_cr.py` (percentile distribution + pooled aggregate CR) |
//! | [`throughput`]                | the perf-counter harness in `paper_benchmark.py` (`encode_speed_mbps` / `decode_speed_mbps`) + `tools/bench_chbmit.py` (`throughput_mibs`) |
//! | [`corpus_summary`]            | the corpus roll-up — delegates to [`crate::harness::run_corpus`], no duplicate logic |
//!
//! What is deliberately NOT here (stays Python, `-m internal`):
//! FSQ entropy / validation, latent utilization, Cayley rotation, residual
//! FSQ, subband leakage — all reach into model internals and have no
//! meaning at the `Codec` byte boundary.

use crate::adapter::{serialize, Codec};
use crate::harness::{self, CorpusSummary};
use crate::metrics;
use std::time::Instant;

/// A corpus file: one `Vec<i64>` per channel plus the sample rate in Hz.
///
/// Type alias for the tuple every battery iterates over, so the signatures
/// read cleanly and the corpus shape is documented in one place.
pub type CorpusFile = (Vec<Vec<i64>>, f64);

/// Flatten a per-channel integer signal into a contiguous `f64` stream,
/// channels concatenated in order.
///
/// The aggregate fidelity metrics (PRD, R) treat the multichannel signal as
/// one flat vector — this matches the Python reference, which calls
/// `.flatten()` before `pearsonr` / the PRD ratio (see `compute_metrics` in
/// `benchmark_rate_distortion.py` and `benchmark_compression_ratio.py`).
fn flatten_f64(signal: &[Vec<i64>]) -> Vec<f64> {
    let total: usize = signal.iter().map(|c| c.len()).sum();
    let mut out = Vec::with_capacity(total);
    for chan in signal {
        out.extend(chan.iter().map(|&s| s as f64));
    }
    out
}

/// Raw (uncompressed) byte size of a signal under the reference container —
/// the denominator of the compression ratio. Mirrors `harness::raw_bytes`:
/// the size the codec is compressing *against*.
fn raw_bytes(signal: &[Vec<i64>]) -> u64 {
    serialize(signal).len() as u64
}

// ===================================================================
// (1) Rate–distortion curve
// ===================================================================

/// One point on a codec's rate–distortion curve for a single file.
///
/// Ports the per-file row produced by `benchmark_rate_distortion.py` /
/// `benchmark_compression_ratio.py`: a `(compression ratio, PRD, R)` triple.
/// In the Python sweep the *rate* axis came from varying FSQ levels; at the
/// `Codec` byte boundary the codec itself sets the operating point, so each
/// file contributes one R-D point and the corpus traces the curve the codec
/// produces across its natural rate spread.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RdPoint {
    /// Compression ratio (raw bytes / compressed bytes) for this file.
    pub cr: f64,
    /// Percentage RMS difference (lower is better). `0.0` when bit-exact.
    pub prd: f64,
    /// Pearson correlation R (higher is better). `1.0` when bit-exact.
    pub r: f64,
}

/// Trace the rate–distortion curve `codec` produces over `signals`.
///
/// Returns one [`RdPoint`] `(cr, prd, r)` per file, in corpus order. This is
/// the curve any codec produces: the Python `benchmark_rate_distortion.py`
/// built it by sweeping FSQ levels and recording `(bps, R, PRD)`; here the
/// codec's own per-file operating point supplies the rate, and we record the
/// matching `(cr, prd, r)`. The metrics are computed on the flattened
/// multichannel signal in the `f64` domain, exactly like the Python
/// `compute_metrics` (flatten, Pearson R, PRD = rms_diff / rms_orig).
///
/// A bit-exact file yields `(cr, 0.0, 1.0)` — zero distortion, perfect
/// correlation — matching the lossless short-circuit in
/// [`crate::harness::run`].
pub fn rate_distortion(codec: &dyn Codec, signals: &[CorpusFile]) -> Vec<RdPoint> {
    let mut points = Vec::with_capacity(signals.len());
    for (signal, fs) in signals {
        let raw = raw_bytes(signal);
        let blob = codec.encode(signal, *fs);
        let comp = blob.len() as u64;
        let recon = codec.decode(&blob);

        let cr = metrics::compression_ratio(raw, comp);

        let orig_f = flatten_f64(signal);
        let recon_f = flatten_f64(&recon);
        let prd = metrics::prd(&orig_f, &recon_f);
        let r = metrics::pearson_r(&orig_f, &recon_f);

        points.push(RdPoint { cr, prd, r });
    }
    points
}

// ===================================================================
// (2) Per-file CR distribution
// ===================================================================

/// Percentile distribution of per-file compression ratios.
///
/// Direct port of the `distribution` block in `tools/bench_per_file_cr.py`:
/// the seven order-statistic percentiles plus the arithmetic mean of the
/// per-file CRs. The percentile index uses the same nearest-rank rule as the
/// Python `pct(p)` helper: `idx = round(p * (n - 1) / 100)`, clamped to
/// `[0, n-1]`, over the ascending-sorted CR list.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CrDistribution {
    /// Number of files contributing to the distribution.
    pub n_files: usize,
    /// Smallest per-file CR.
    pub min: f64,
    /// 5th percentile CR.
    pub p5: f64,
    /// 25th percentile CR.
    pub p25: f64,
    /// Median (50th percentile) CR.
    pub median: f64,
    /// 75th percentile CR.
    pub p75: f64,
    /// 95th percentile CR.
    pub p95: f64,
    /// Largest per-file CR.
    pub max: f64,
    /// Arithmetic mean of the per-file CRs.
    pub mean: f64,
}

/// Per-file compression-ratio distribution for `codec` over `signals`.
///
/// For each file, encode and form `cr = raw / compressed`, then summarize
/// the spread with the order-statistic percentiles + mean from
/// `bench_per_file_cr.py`. This is the per-file CR claim the paper's §IV.C
/// substantiates: the range from near-1:1 on noisy recordings up to
/// dozens:1 on low-noise segments, made auditable as fixed percentiles.
///
/// An empty corpus returns an all-zero distribution with `n_files == 0`
/// (the Python tool returns `{"files": 0}` and skips the distribution; the
/// all-zero record is the typed equivalent).
///
/// Note: this is the *unpooled* per-file spread. The single pooled aggregate
/// CR (byte-weighted) is reported by [`corpus_summary`] — the two answer
/// different questions and are intentionally separate.
pub fn per_file_cr_distribution(codec: &dyn Codec, signals: &[CorpusFile]) -> CrDistribution {
    let n = signals.len();
    if n == 0 {
        return CrDistribution {
            n_files: 0,
            min: 0.0,
            p5: 0.0,
            p25: 0.0,
            median: 0.0,
            p75: 0.0,
            p95: 0.0,
            max: 0.0,
            mean: 0.0,
        };
    }

    let mut crs: Vec<f64> = Vec::with_capacity(n);
    for (signal, fs) in signals {
        let raw = raw_bytes(signal);
        let comp = codec.encode(signal, *fs).len() as u64;
        crs.push(metrics::compression_ratio(raw, comp));
    }

    // Ascending sort for the order statistics. CR values are finite (raw is
    // finite, the divisor is clamped to >= 1 by compression_ratio), so a
    // total order via partial_cmp is well-defined here.
    crs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mean = crs.iter().sum::<f64>() / n as f64;

    // Nearest-rank percentile, matching bench_per_file_cr.py's `pct`:
    //   idx = clamp(round(p * (n - 1) / 100), 0, n - 1)
    let pct = |p: f64| -> f64 {
        let idx = (p * (n as f64 - 1.0) / 100.0).round() as i64;
        let idx = idx.clamp(0, n as i64 - 1) as usize;
        crs[idx]
    };

    CrDistribution {
        n_files: n,
        min: crs[0],
        p5: pct(5.0),
        p25: pct(25.0),
        median: pct(50.0),
        p75: pct(75.0),
        p95: pct(95.0),
        max: crs[n - 1],
        mean,
    }
}

// ===================================================================
// (3) Throughput
// ===================================================================

/// Encode / decode throughput in MiB/s over a corpus.
///
/// Ports the perf-counter harness from `paper_benchmark.py`
/// (`encode_speed_mbps` / `decode_speed_mbps`) and `tools/bench_chbmit.py`
/// (`throughput_mibs`): wall-clock the encode pass and the decode pass
/// separately, then divide the *raw* corpus byte total by each elapsed time.
/// Rates are reported in MiB/s (`1024 * 1024` bytes), matching the
/// `bench_chbmit.py` convention (`paper_benchmark.py`'s `MB/s` used `1e6`;
/// we standardize on the harness's MiB to stay consistent with
/// [`crate::report::LqsReport::throughput_mibs`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Throughput {
    /// Total raw (uncompressed) bytes processed across the corpus.
    pub raw_bytes: u64,
    /// Encode throughput in MiB/s over the raw payload.
    pub encode_mibs: f64,
    /// Decode throughput in MiB/s over the raw payload.
    pub decode_mibs: f64,
}

/// Measure encode + decode throughput of `codec` over `signals`.
///
/// Two timed passes over the whole corpus — all encodes, then all decodes —
/// so the encode and decode rates are independent (mirrors the separate
/// `encode_speed_mbps` / `decode_speed_mbps` in `paper_benchmark.py`). The
/// blobs from the encode pass are retained and fed to the decode pass so the
/// decode timing measures real reconstruction, not re-encoding.
///
/// Throughput is `raw_total / elapsed`, where `raw_total` is the pooled raw
/// byte size of the corpus. A zero / sub-tick elapsed time (tiny synthetic
/// fixtures) yields a `0.0` rate rather than `+inf`, matching the guard in
/// [`crate::harness::run`].
pub fn throughput(codec: &dyn Codec, signals: &[CorpusFile]) -> Throughput {
    let raw_total: u64 = signals.iter().map(|(s, _)| raw_bytes(s)).sum();

    // ── Encode pass: time every encode, keep the blobs. ──────────────
    let t_enc = Instant::now();
    let mut blobs: Vec<Vec<u8>> = Vec::with_capacity(signals.len());
    for (signal, fs) in signals {
        blobs.push(codec.encode(signal, *fs));
    }
    let enc_secs = t_enc.elapsed().as_secs_f64();

    // ── Decode pass: time every decode of the retained blobs. ─────────
    let t_dec = Instant::now();
    for blob in &blobs {
        let _ = codec.decode(blob);
    }
    let dec_secs = t_dec.elapsed().as_secs_f64();

    let mib = raw_total as f64 / (1024.0 * 1024.0);
    let encode_mibs = if enc_secs > 0.0 { mib / enc_secs } else { 0.0 };
    let decode_mibs = if dec_secs > 0.0 { mib / dec_secs } else { 0.0 };

    Throughput {
        raw_bytes: raw_total,
        encode_mibs,
        decode_mibs,
    }
}

// ===================================================================
// (4) Corpus summary
// ===================================================================

/// Aggregate corpus roll-up for `codec` over `signals`.
///
/// The byte-pooled aggregate CR, mean PRD/R, and worst grade across the
/// corpus. This logic already lives in [`crate::harness::run_corpus`] (it
/// pools CR by total bytes via [`crate::metrics::aggregate_cr`], averages
/// PRD/R per file, and tracks the weakest grade), so this battery is a thin
/// re-export that drops the per-file reports and returns only the
/// [`CorpusSummary`] roll-up — no duplicated aggregation.
pub fn corpus_summary(codec: &dyn Codec, signals: &[CorpusFile]) -> CorpusSummary {
    let (_reports, summary) = harness::run_corpus(codec, signals);
    summary
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::{Gzip, Store};

    /// A deterministic, repeatable synthetic signal: a small multichannel
    /// integer waveform with cross-band energy. Reused across batteries so
    /// the corpus is fixed and the assertions are exact.
    fn synthetic(scale: f64) -> Vec<i64> {
        let fs = 256.0;
        let n = 256;
        (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                let v = scale
                    * (100.0 * (2.0 * std::f64::consts::PI * 2.0 * t).sin()
                        + 60.0 * (2.0 * std::f64::consts::PI * 10.0 * t).sin()
                        + 30.0 * (2.0 * std::f64::consts::PI * 40.0 * t).sin());
                v.round() as i64
            })
            .collect()
    }

    /// A three-file corpus of distinct synthetic signals at 256 Hz.
    fn corpus() -> Vec<CorpusFile> {
        vec![
            (vec![synthetic(1.0), synthetic(1.5)], 256.0),
            (vec![synthetic(0.7), synthetic(2.0), synthetic(1.2)], 256.0),
            (vec![synthetic(0.5)], 256.0),
        ]
    }

    // ── rate_distortion ──────────────────────────────────────────────

    #[test]
    fn rd_lossless_is_zero_distortion() {
        // Identical reconstruction => PRD exactly 0, R exactly 1, for every
        // file, under both lossless reference adapters.
        let files = corpus();
        for pts in [rate_distortion(&Store, &files), rate_distortion(&Gzip, &files)] {
            assert_eq!(pts.len(), files.len());
            for p in &pts {
                assert_eq!(p.prd, 0.0, "lossless codec must have prd 0");
                assert_eq!(p.r, 1.0, "lossless codec must have r 1");
                assert!(p.cr > 0.0 && p.cr.is_finite());
            }
        }
    }

    #[test]
    fn rd_gzip_compresses_more_than_store() {
        // Store's blob IS the serialization (cr ~= 1.0); gzip squeezes the
        // structured synthetic signal further (cr > store's).
        let files = corpus();
        let store = rate_distortion(&Store, &files);
        let gzip = rate_distortion(&Gzip, &files);
        for (s, g) in store.iter().zip(&gzip) {
            assert!(
                g.cr >= s.cr,
                "gzip cr {} should be >= store cr {}",
                g.cr,
                s.cr
            );
        }
    }

    #[test]
    fn rd_deterministic() {
        let files = corpus();
        assert_eq!(rate_distortion(&Store, &files), rate_distortion(&Store, &files));
    }

    #[test]
    fn rd_empty_corpus() {
        assert!(rate_distortion(&Store, &[]).is_empty());
    }

    // ── per_file_cr_distribution ─────────────────────────────────────

    #[test]
    fn cr_distribution_ordered_and_consistent() {
        let files = corpus();
        let d = per_file_cr_distribution(&Gzip, &files);
        assert_eq!(d.n_files, files.len());
        // Percentiles are monotone non-decreasing.
        assert!(d.min <= d.p5);
        assert!(d.p5 <= d.p25);
        assert!(d.p25 <= d.median);
        assert!(d.median <= d.p75);
        assert!(d.p75 <= d.p95);
        assert!(d.p95 <= d.max);
        // Mean lies within [min, max].
        assert!(d.mean >= d.min && d.mean <= d.max);
    }

    #[test]
    fn cr_distribution_percentile_indices() {
        // Single file: every percentile collapses to that file's CR.
        let one = vec![(vec![synthetic(1.0)], 256.0)];
        let d = per_file_cr_distribution(&Gzip, &one);
        assert_eq!(d.n_files, 1);
        assert_eq!(d.min, d.max);
        assert_eq!(d.median, d.min);
        assert_eq!(d.mean, d.min);
        assert_eq!(d.p5, d.min);
        assert_eq!(d.p95, d.min);
    }

    #[test]
    fn cr_distribution_store_is_near_one() {
        // Store does not compress: every per-file CR ~= 1.0, so the whole
        // distribution sits near 1.0.
        let files = corpus();
        let d = per_file_cr_distribution(&Store, &files);
        assert!(d.min >= 0.8 && d.max <= 1.2, "store cr spread {:?}", d);
    }

    #[test]
    fn cr_distribution_deterministic() {
        let files = corpus();
        assert_eq!(
            per_file_cr_distribution(&Gzip, &files),
            per_file_cr_distribution(&Gzip, &files)
        );
    }

    #[test]
    fn cr_distribution_empty_corpus() {
        let d = per_file_cr_distribution(&Store, &[]);
        assert_eq!(d.n_files, 0);
        assert_eq!(d.mean, 0.0);
        assert_eq!(d.median, 0.0);
    }

    // ── throughput ───────────────────────────────────────────────────

    #[test]
    fn throughput_reports_finite_rates() {
        let files = corpus();
        let t = throughput(&Gzip, &files);
        let expected_raw: u64 = files
            .iter()
            .map(|(s, _)| serialize(s).len() as u64)
            .sum();
        assert_eq!(t.raw_bytes, expected_raw);
        assert!(t.encode_mibs.is_finite());
        assert!(t.decode_mibs.is_finite());
        assert!(t.encode_mibs >= 0.0);
        assert!(t.decode_mibs >= 0.0);
    }

    #[test]
    fn throughput_empty_corpus_is_zero() {
        let t = throughput(&Store, &[]);
        assert_eq!(t.raw_bytes, 0);
        // No bytes + no elapsed => guarded 0.0, never +inf.
        assert_eq!(t.encode_mibs, 0.0);
        assert_eq!(t.decode_mibs, 0.0);
    }

    // ── corpus_summary ───────────────────────────────────────────────

    #[test]
    fn corpus_summary_matches_run_corpus() {
        // The battery must be a faithful thin wrapper over run_corpus.
        let files = corpus();
        let summary = corpus_summary(&Store, &files);
        let (_, expected) = harness::run_corpus(&Store, &files);
        assert_eq!(summary.codec, expected.codec);
        assert_eq!(summary.n_files, expected.n_files);
        assert_eq!(summary.mean_cr, expected.mean_cr);
        assert_eq!(summary.mean_prd, expected.mean_prd);
        assert_eq!(summary.mean_r, expected.mean_r);
        assert_eq!(summary.worst_grade, expected.worst_grade);
        assert_eq!(summary.all_bit_exact, expected.all_bit_exact);
    }

    #[test]
    fn corpus_summary_lossless_is_grade_l() {
        let files = corpus();
        let summary = corpus_summary(&Gzip, &files);
        assert_eq!(summary.n_files, files.len());
        assert!(summary.all_bit_exact, "gzip is lossless on synthetic corpus");
        assert_eq!(summary.worst_grade, 'L');
        assert_eq!(summary.mean_prd, 0.0);
        assert_eq!(summary.mean_r, 1.0);
        // Gzip compresses the structured signal: aggregate CR clears 1.0.
        assert!(summary.mean_cr >= 1.0);
    }

    #[test]
    fn corpus_summary_empty_is_below_floor() {
        let summary = corpus_summary(&Store, &[]);
        assert_eq!(summary.n_files, 0);
        assert_eq!(summary.worst_grade, '\0');
    }
}
