//! Integration tests — the LQS standard, exercised end-to-end.
//!
//! These tests drive the public crate surface exactly as a third party
//! would: build a signal, run a codec through [`lqs::harness::run`], and
//! assert the verdict. They prove the four load-bearing guarantees of the
//! standard:
//!
//! 1. A lossless passthrough codec (`Store`) grades `'L'` on any signal.
//! 2. A compressing lossless codec (`Gzip`) grades `'L'` even when its CR
//!    is modest — losslessness is decided by bit-exactness, not ratio
//!    (subject only to the `cr >= 0.8` no-expansion floor).
//! 3. A deliberately-lossy codec is NEVER graded `'L'`: it lands in
//!    {C, M, A, below-floor} according to its actual PRD / R / CR.
//! 4. The L-tier gate is on the INTEGER sample domain: a float-PRD path
//!    that is "zero to ~1e-12" does not falsely pass exact-zero.

use std::f64::consts::PI;

use lqs::adapter::{deserialize, serialize, Codec, Gzip, Store};
use lqs::harness;
use lqs::metrics;

/// Deterministic synthetic multichannel signal with cross-band energy.
fn signal(n_chan: usize, n: usize, fs: f64) -> Vec<Vec<i64>> {
    (0..n_chan)
        .map(|c| {
            let amp = 1.0 + 0.25 * c as f64;
            (0..n)
                .map(|i| {
                    let t = i as f64 / fs;
                    let v = amp
                        * (50.0
                            + 120.0 * (2.0 * PI * 2.0 * t).sin()
                            + 70.0 * (2.0 * PI * 10.0 * t).sin()
                            + 25.0 * (2.0 * PI * 40.0 * t).sin());
                    v.round() as i64
                })
                .collect()
        })
        .collect()
}

/// A deliberately-lossy codec: integer-divide by `step` on encode, multiply
/// back on decode. Discards the low-order bits, so it is not bit-exact.
struct Quantize {
    step: i64,
}

impl Codec for Quantize {
    fn name(&self) -> &str {
        "quantize"
    }
    fn declared_lossless(&self) -> bool {
        false
    }
    fn encode(&self, sig: &[Vec<i64>], _fs: f64) -> Vec<u8> {
        let q: Vec<Vec<i64>> = sig
            .iter()
            .map(|c| c.iter().map(|&s| s / self.step).collect())
            .collect();
        serialize(&q)
    }
    fn decode(&self, blob: &[u8]) -> Vec<Vec<i64>> {
        deserialize(blob)
            .into_iter()
            .map(|c| c.into_iter().map(|s| s * self.step).collect())
            .collect()
    }
}

#[test]
fn store_grades_lossless_on_any_signal() {
    // Several shapes, including a degenerate single-channel and a wide one.
    for (nc, n) in [(1usize, 64usize), (4, 256), (8, 512)] {
        let sig = signal(nc, n, 256.0);
        let rep = harness::run(&Store, &sig, 256.0);
        assert!(rep.bit_exact, "store {nc}x{n}: not bit-exact");
        assert_eq!(rep.grade, 'L', "store {nc}x{n}: grade != L");
        assert_eq!(rep.prd, 0.0);
        assert_eq!(rep.r, 1.0);
        assert!(rep.passed());
    }
}

#[test]
fn gzip_grades_lossless_even_with_modest_cr() {
    let sig = signal(4, 256, 256.0);
    let rep = harness::run(&Gzip, &sig, 256.0);
    assert!(rep.bit_exact, "gzip must be bit-exact");
    assert_eq!(rep.grade, 'L', "gzip lossless must grade L regardless of CR");
    // CR is whatever miniz_oxide achieves; losslessness does not depend on
    // it (only on clearing the 0.8 no-expansion floor).
    assert!(rep.cr >= 0.8, "gzip cr {} below L floor", rep.cr);
}

#[test]
fn lossy_codec_is_never_graded_lossless() {
    let sig = signal(4, 256, 256.0);
    let rep = harness::run(&Quantize { step: 8 }, &sig, 256.0);

    // The defining guarantee: a lossy codec never sneaks into L.
    assert!(!rep.bit_exact, "quantize must NOT be bit-exact");
    assert_ne!(rep.grade, 'L', "lossy codec must never grade L");
    // It must land in one of the lossy tiers or below the floor.
    assert!(
        matches!(rep.grade, 'C' | 'M' | 'A' | '\0'),
        "unexpected grade {:?}",
        rep.grade
    );
    // And it must have measured some distortion (PRD > 0) — the ÷8/×8
    // round trip truncates the low 3 bits of magnitude.
    assert!(rep.prd > 0.0, "lossy codec should report nonzero PRD");
    // The grade the harness assigned must be self-consistent: re-grading
    // the report's own metrics reproduces it.
    let band_triples: Vec<(String, f64, f64)> = rep
        .per_band
        .iter()
        .map(|b| (b.band.clone(), b.r, b.prd))
        .collect();
    let regraded = lqs::levels::grade(rep.r, rep.prd, rep.cr, rep.snr_db, &band_triples);
    assert_eq!(
        regraded.grade, rep.grade,
        "harness grade must match re-grading its own metrics"
    );
}

#[test]
fn heavier_quantization_degrades_or_fails() {
    // A coarser step injects more distortion than a fine one. The coarse
    // codec's PRD must be >= the fine codec's, and neither may be 'L'.
    let sig = signal(4, 256, 256.0);
    let fine = harness::run(&Quantize { step: 4 }, &sig, 256.0);
    let coarse = harness::run(&Quantize { step: 64 }, &sig, 256.0);
    assert_ne!(fine.grade, 'L');
    assert_ne!(coarse.grade, 'L');
    assert!(
        coarse.prd >= fine.prd,
        "coarser quantization should not have lower PRD ({} < {})",
        coarse.prd,
        fine.prd
    );
}

#[test]
fn exact_zero_gate_is_integer_domain_not_float() {
    // The L-tier gate is `prd_is_exact_zero` on i64 samples, NOT a float
    // PRD compared against an epsilon. Demonstrate the distinction:
    //
    // Two integer streams differing by one LSB are NOT exact-zero, even
    // though their float PRD is ~1e-12-small. A naive `float_prd < eps`
    // gate would wrongly pass them as lossless; the integer gate rejects.
    let orig: Vec<i64> = (0..512).map(|i| 1_000_000 + i).collect();
    let mut perturbed = orig.clone();
    *perturbed.last_mut().unwrap() += 1; // single LSB off, last sample

    // Integer-domain gate: correctly NOT exact.
    assert!(
        !metrics::prd_is_exact_zero(&orig, &perturbed),
        "one-LSB difference must not be exact-zero"
    );

    // The float PRD of this near-identical pair is vanishingly small —
    // small enough that an epsilon gate would have falsely accepted it.
    let of: Vec<f64> = orig.iter().map(|&x| x as f64).collect();
    let pf: Vec<f64> = perturbed.iter().map(|&x| x as f64).collect();
    let float_prd = metrics::prd(&of, &pf);
    assert!(float_prd > 0.0, "perturbed pair has nonzero float PRD");
    assert!(
        float_prd < 1e-3,
        "float PRD {float_prd} should be tiny — the trap a float gate falls into"
    );

    // Identical integer streams ARE exact-zero (the honest lossless case).
    assert!(metrics::prd_is_exact_zero(&orig, &orig));
}

#[test]
fn lossy_codec_does_not_falsely_pass_lossless_via_float_roundoff() {
    // End-to-end version of the integer-domain guarantee: drive the harness
    // with a codec whose output is off by one LSB on a single sample (a
    // "nearly lossless" codec). It must report bit_exact=false and grade
    // anything but 'L' — the integer gate is not fooled by a near-zero
    // float PRD.
    struct OffByOne;
    impl Codec for OffByOne {
        fn name(&self) -> &str {
            "off-by-one"
        }
        fn declared_lossless(&self) -> bool {
            true // it *claims* lossless — the harness must verify, not trust
        }
        fn encode(&self, sig: &[Vec<i64>], _fs: f64) -> Vec<u8> {
            serialize(sig)
        }
        fn decode(&self, blob: &[u8]) -> Vec<Vec<i64>> {
            let mut sig = deserialize(blob);
            // Perturb exactly one sample by one LSB.
            if let Some(chan) = sig.iter_mut().find(|c| !c.is_empty()) {
                chan[0] += 1;
            }
            sig
        }
    }

    let sig = signal(4, 256, 256.0);
    let rep = harness::run(&OffByOne, &sig, 256.0);
    assert!(
        !rep.bit_exact,
        "a one-LSB-off codec must not be reported bit-exact despite its claim"
    );
    assert_ne!(rep.grade, 'L', "claimed-lossless-but-not must not grade L");
}

#[test]
fn report_serializes_round_trip() {
    // The standard's exchange format: a report serializes to JSON and back
    // without losing the verdict.
    let sig = signal(2, 128, 256.0);
    let rep = harness::run(&Store, &sig, 256.0);
    let json = rep.to_json();
    let back = lqs::report::LqsReport::from_json(&json).expect("report JSON round-trips");
    assert_eq!(back, rep);
    assert_eq!(back.grade, 'L');
}

#[test]
fn corpus_run_rolls_up() {
    let files = vec![
        (signal(2, 128, 256.0), 256.0),
        (signal(4, 256, 256.0), 256.0),
    ];
    let (reports, summary) = harness::run_corpus(&Gzip, &files);
    assert_eq!(reports.len(), 2);
    assert_eq!(summary.n_files, 2);
    assert!(summary.all_bit_exact);
    assert_eq!(summary.worst_grade, 'L');
}
