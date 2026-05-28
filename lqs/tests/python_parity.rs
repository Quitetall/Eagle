//! Python ↔ Rust tier-table parity contract.
//!
//! The Rust tier table in [`lqs::levels`] is the canonical LQS spec (see
//! the crate docs: Rust is what the grading gate enforces). The Python
//! reference at
//! `LamQuant-Lossless/reference_implementations/python_codec/lamquant_codec/lqs.py`
//! (`LQS_LEVELS` dict) is a readable mirror. This test pins the SHARED
//! lossy tiers C / M / A so that drift in EITHER implementation trips CI.
//!
//! ## What is asserted
//!
//! The Python C / M / A thresholds and per-band requirements, copied
//! verbatim from the Python `LQS_LEVELS` dict, are hard-coded below as the
//! contract. We then assert that the Rust [`lqs::levels::levels()`] table
//! matches them field-for-field:
//!
//!   - global `max_prd`, `min_r`, `max_snr_loss`, `min_cr`
//!   - every per-band `(freq_range, max_prd, min_r)` triple
//!
//! At the time of writing the Rust and Python C / M / A values agree
//! exactly (verified by hand against both source files), so the contract
//! values below double as the authoritative Python snapshot.
//!
//! ## The one allowed divergence: tier L
//!
//! The L (Lossless) tier intentionally DIFFERS between the two and is NOT
//! asserted for equality here:
//!
//!   - Rust : L is an exact-zero PRD short-circuit (`max_prd == 0.0`,
//!            `min_r == 1.0`) with the vendor-neutral no-expansion CR floor
//!            `min_cr == 0.8`.
//!   - Python: `LQS_LEVELS['L']` carries `min_cr == 2.0` and leans on the
//!            `bit_exact` flag rather than a CR floor.
//!
//! This divergence is by design (Rust grades losslessness on the integer
//! sample domain and only forbids expansion; Python's old `min_cr=2.0`
//! would refuse to certify a bit-exact-but-barely-compressed file as
//! lossless). The test below documents it explicitly and asserts the
//! Rust L-tier invariants on their own terms — it does NOT require the
//! two L definitions to match.

use lqs::levels::{self, LqsLevel};

/// The Python-side contract for one band: `(freq_lo, freq_hi, max_prd, min_r)`.
type BandContract = (f64, f64, f64, f64);

/// The Python-side contract for one lossy tier.
struct TierContract {
    code: char,
    name: &'static str,
    max_prd: f64,
    min_r: f64,
    max_snr_loss: f64,
    min_cr: f64,
    /// (band_name, freq_lo, freq_hi, max_prd, min_r), copied from Python.
    bands: &'static [(&'static str, BandContract)],
}

/// C / M / A as written in the Python `LQS_LEVELS` dict. These are the
/// authoritative shared thresholds; the Rust table must equal them.
const PYTHON_CMA: &[TierContract] = &[
    // 'C': Clinical — max_prd=9.0, min_r=0.95, max_snr_loss=3.0, min_cr=20.0
    TierContract {
        code: 'C',
        name: "Clinical",
        max_prd: 9.0,
        min_r: 0.95,
        max_snr_loss: 3.0,
        min_cr: 20.0,
        bands: &[
            ("delta", (0.5, 4.0, 5.0, 0.98)),
            ("theta", (4.0, 8.0, 7.0, 0.97)),
            ("alpha", (8.0, 13.0, 8.0, 0.96)),
            ("beta", (13.0, 30.0, 12.0, 0.93)),
            ("gamma", (30.0, 50.0, 20.0, 0.85)),
        ],
    },
    // 'M': Monitoring — max_prd=20.0, min_r=0.85, max_snr_loss=6.0, min_cr=100.0
    TierContract {
        code: 'M',
        name: "Monitoring",
        max_prd: 20.0,
        min_r: 0.85,
        max_snr_loss: 6.0,
        min_cr: 100.0,
        bands: &[
            ("delta", (0.5, 4.0, 10.0, 0.95)),
            ("theta", (4.0, 8.0, 12.0, 0.93)),
            ("alpha", (8.0, 13.0, 15.0, 0.90)),
            ("beta", (13.0, 30.0, 25.0, 0.80)),
            ("gamma", (30.0, 50.0, 40.0, 0.60)),
        ],
    },
    // 'A': Alerting — max_prd=40.0, min_r=0.70, max_snr_loss=10.0, min_cr=200.0
    TierContract {
        code: 'A',
        name: "Alerting",
        max_prd: 40.0,
        min_r: 0.70,
        max_snr_loss: 10.0,
        min_cr: 200.0,
        bands: &[
            ("delta", (0.5, 4.0, 20.0, 0.85)),
            ("theta", (4.0, 8.0, 25.0, 0.80)),
            ("alpha", (8.0, 13.0, 30.0, 0.75)),
            ("beta", (13.0, 30.0, 40.0, 0.65)),
            ("gamma", (30.0, 50.0, 60.0, 0.40)),
        ],
    },
];

/// Exact f64 equality is what we want here: these are spec constants typed
/// out in both files, not the result of arithmetic, so any difference is a
/// real drift and must fail.
fn assert_eq_f64(label: &str, rust: f64, python: f64) {
    assert!(
        rust == python,
        "{label}: Rust {rust} != Python {python} — tier table drifted; \
         reconcile lqs/src/levels.rs against the Python LQS_LEVELS dict",
    );
}

fn rust_tier(code: char) -> LqsLevel {
    levels::level_by_char(code).unwrap_or_else(|| panic!("Rust table missing tier {code:?}"))
}

#[test]
fn cma_tiers_match_python_globals() {
    for c in PYTHON_CMA {
        let r = rust_tier(c.code);
        assert_eq!(r.name, c.name, "tier {} name", c.code);
        assert_eq_f64(&format!("tier {} max_prd", c.code), r.max_prd, c.max_prd);
        assert_eq_f64(&format!("tier {} min_r", c.code), r.min_r, c.min_r);
        assert_eq_f64(
            &format!("tier {} max_snr_loss", c.code),
            r.max_snr_loss,
            c.max_snr_loss,
        );
        assert_eq_f64(&format!("tier {} min_cr", c.code), r.min_cr, c.min_cr);
    }
}

#[test]
fn cma_tiers_match_python_per_band() {
    for c in PYTHON_CMA {
        let r = rust_tier(c.code);

        // Same set of band names, same count.
        assert_eq!(
            r.band_fidelity.len(),
            c.bands.len(),
            "tier {} band count",
            c.code
        );

        for (band_name, (lo, hi, max_prd, min_r)) in c.bands {
            let rb = r
                .band_fidelity
                .get(*band_name)
                .unwrap_or_else(|| panic!("tier {} missing band {band_name}", c.code));
            assert_eq_f64(
                &format!("tier {} band {band_name} freq_lo", c.code),
                rb.freq_range.0,
                *lo,
            );
            assert_eq_f64(
                &format!("tier {} band {band_name} freq_hi", c.code),
                rb.freq_range.1,
                *hi,
            );
            assert_eq_f64(
                &format!("tier {} band {band_name} max_prd", c.code),
                rb.max_prd,
                *max_prd,
            );
            assert_eq_f64(
                &format!("tier {} band {band_name} min_r", c.code),
                rb.min_r,
                *min_r,
            );
        }
    }
}

/// The L tier is the one ALLOWED divergence. We do NOT assert it equals
/// Python; instead we pin the Rust L invariants so the vendor-neutral
/// lossless definition itself cannot silently drift.
///
/// Rust L: exact-zero PRD short-circuit, min_r == 1.0, min_cr == 0.8
/// (no-expansion floor), no per-band requirements.
///
/// Python L (for the record, intentionally different): min_cr == 2.0 and
/// a `bit_exact` flag. If the Python file is ever revised to adopt the
/// vendor-neutral 0.8 floor, update the comment here — but it is not
/// required for parity.
#[test]
fn l_tier_is_the_documented_divergence() {
    let l = rust_tier('L');
    assert_eq!(l.name, "Lossless");
    assert_eq_f64("L max_prd", l.max_prd, 0.0);
    assert_eq_f64("L min_r", l.min_r, 1.0);
    assert_eq_f64("L min_cr (vendor-neutral no-expansion floor)", l.min_cr, 0.8);
    assert!(
        l.band_fidelity.is_empty(),
        "L tier must carry no per-band requirements"
    );

    // Document the divergence explicitly: Rust 0.8 vs Python 2.0.
    const PYTHON_L_MIN_CR: f64 = 2.0;
    assert!(
        l.min_cr != PYTHON_L_MIN_CR,
        "L tier is expected to DIVERGE from Python (Rust 0.8 vs Python 2.0); \
         if they now agree, this divergence note is stale and should be revisited"
    );
}

/// Sanity: the full Rust table is exactly the four tiers in strictness
/// order. Guards against an accidental extra/missing tier sneaking past
/// the per-tier assertions above.
#[test]
fn rust_table_is_lcma_in_order() {
    let t = levels::levels();
    let codes: Vec<char> = t.iter().map(|l| l.level).collect();
    assert_eq!(codes, vec!['L', 'C', 'M', 'A'], "tier order / membership");
}
