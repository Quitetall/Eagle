//! Numerical value-parity contract: Rust `lqs::metrics` ↔ Python reference.
//!
//! This is the NUMERICAL counterpart to `python_parity.rs` (which pins the
//! tier *table*). Here we pin the actual metric *values* the Rust code
//! computes against frozen golden fixtures emitted by the canonical Python
//! reference.
//!
//! ## Provenance
//!
//! `tests/golden/metrics_parity.json` is generated ONCE by
//! `tools/parity/gen_golden.py`, which imports the real Python functions and
//! freezes their output:
//!
//!   - `py_prd`        ← `ai_models.metrics.prd_numpy`
//!   - `py_pearson_r`  ← `ai_models.metrics.pearson_r_numpy` (single-pass;
//!                        bit-identical to the Rust formula on these fixtures)
//!   - `py_snr_db`     ← `lamquant_codec.lqs.snr_db`
//!   - `py_prdn`       ← numpy replication of the Rust `prdn` formula
//!                        (no Python reference exists for normalized PRD)
//!   - `py_entropy_bits` ← numpy replication of `entropy_from_counts`
//!   - `py_cr`         ← numpy replication of `compression_ratio`
//!   - `py_prd_lqs` / `py_pearson_r_lqs` ← `lamquant_codec.lqs.{prd,pearson_r}`
//!                        recorded as a CROSS-CHECK (a second Python impl)
//!
//! This test is PURE RUST — it never invokes Python/torch. The golden JSON is
//! the frozen contract, matching the project's "pin against frozen golden
//! fixtures" testing principle.
//!
//! ## Tolerance
//!
//! `|rust - py| < 1e-9` for ordinary magnitudes, switching to a relative
//! `|rust - py| / |py| < 1e-9` when `|py| >= 1.0` so large SNR/PRD values are
//! not held to an unrealistic absolute floor. Non-finite golden values
//! (`-inf` for the all-zero-original SNR guard) require exact bit equality —
//! both Python and Rust genuinely produce `-inf` there, so we assert it
//! rather than skip it.
//!
//! If a metric DIVERGES it is collected and reported at the end with metric
//! name + Rust value + Python value; the tolerance is NEVER loosened to hide
//! a real divergence.

use std::fs;
use std::path::PathBuf;

use serde::Deserialize;

use lqs::metrics;

/// A float that may be a JSON number or a sentinel string for non-finite
/// values (`"inf"`, `"-inf"`, `"nan"`). The Python generator emits the string
/// form because `serde_json` (and JSON proper) reject bare `Infinity`/`NaN`.
#[derive(Debug, Clone, Copy)]
struct MaybeFloat(f64);

impl<'de> Deserialize<'de> for MaybeFloat {
    fn deserialize<D>(d: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Raw {
            Num(f64),
            Str(String),
        }
        let raw = Raw::deserialize(d)?;
        let v = match raw {
            Raw::Num(n) => n,
            Raw::Str(s) => match s.as_str() {
                "inf" | "Infinity" | "+inf" => f64::INFINITY,
                "-inf" | "-Infinity" => f64::NEG_INFINITY,
                "nan" | "NaN" => f64::NAN,
                other => {
                    return Err(serde::de::Error::custom(format!(
                        "unrecognized float sentinel string: {other:?}"
                    )))
                }
            },
        };
        Ok(MaybeFloat(v))
    }
}

#[derive(Debug, Deserialize)]
struct Fixture {
    name: String,
    #[allow(dead_code)]
    #[serde(default)]
    note: String,
    orig: Vec<f64>,
    recon: Vec<f64>,

    py_prd: MaybeFloat,
    py_prdn: MaybeFloat,
    py_pearson_r: MaybeFloat,
    py_snr_db: MaybeFloat,

    // Cross-check columns (second Python implementation). Pinned with a
    // looser-but-still-tight tolerance because np.corrcoef differs from the
    // single-pass formula by ~1e-16.
    py_prd_lqs: MaybeFloat,
    py_pearson_r_lqs: MaybeFloat,

    // Optional metrics present only on fixtures that carry the inputs.
    #[serde(default)]
    counts: Option<Vec<u64>>,
    #[serde(default)]
    py_entropy_bits: Option<MaybeFloat>,
    #[serde(default)]
    raw_bytes: Option<u64>,
    #[serde(default)]
    comp_bytes: Option<u64>,
    #[serde(default)]
    py_cr: Option<MaybeFloat>,
}

#[derive(Debug, Deserialize)]
struct Golden {
    #[serde(rename = "_numpy_version")]
    #[allow(dead_code)]
    numpy_version: Option<String>,
    fixtures: Vec<Fixture>,
}

/// Absolute / relative tolerance for the canonical metrics.
const TOL: f64 = 1e-9;
/// The cross-check Python implementation (np.corrcoef) differs from the
/// single-pass formula by ~1e-16; pin it a bit looser but still extremely
/// tight, so a real drift in EITHER Python impl still trips.
const CROSS_TOL: f64 = 1e-9;

/// One recorded mismatch.
struct Divergence {
    fixture: String,
    metric: String,
    rust: f64,
    python: f64,
}

/// Compare a Rust-computed value against a Python golden value.
///
/// - Non-finite golden values require exact bit equality (both sides must be
///   the same flavour of inf, or both NaN).
/// - Finite values use `|r - p| < tol`, switching to relative when
///   `|p| >= 1.0`.
fn within(rust: f64, python: f64, tol: f64) -> bool {
    if !python.is_finite() || !rust.is_finite() {
        // -inf == -inf, +inf == +inf, NaN matches NaN. Anything else fails.
        if python.is_nan() || rust.is_nan() {
            return python.is_nan() && rust.is_nan();
        }
        return rust == python;
    }
    let abs = (rust - python).abs();
    if python.abs() >= 1.0 {
        abs / python.abs() < tol || abs < tol
    } else {
        abs < tol
    }
}

fn check(
    out: &mut Vec<Divergence>,
    fixture: &str,
    metric: &str,
    rust: f64,
    python: f64,
    tol: f64,
) {
    if !within(rust, python, tol) {
        out.push(Divergence {
            fixture: fixture.to_string(),
            metric: metric.to_string(),
            rust,
            python,
        });
    }
}

fn golden_path() -> PathBuf {
    // CARGO_MANIFEST_DIR == .../Eagle/lqs at test time.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("golden")
        .join("metrics_parity.json")
}

fn load_golden() -> Golden {
    let path = golden_path();
    let raw = fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "failed to read golden fixtures at {}: {e}\n\
             regenerate with: cd Eagle && PYTHONPATH=<neural>:<lossless> \
             python3 tools/parity/gen_golden.py",
            path.display()
        )
    });
    serde_json::from_str(&raw)
        .unwrap_or_else(|e| panic!("failed to parse {}: {e}", path.display()))
}

#[test]
fn metrics_match_python_golden() {
    let golden = load_golden();
    assert!(
        !golden.fixtures.is_empty(),
        "golden fixture list is empty — regenerate metrics_parity.json"
    );

    let mut divergences: Vec<Divergence> = Vec::new();

    for fx in &golden.fixtures {
        // Canonical metrics — Rust must match the canonical Python column.
        let r_prd = metrics::prd(&fx.orig, &fx.recon);
        check(&mut divergences, &fx.name, "prd", r_prd, fx.py_prd.0, TOL);

        let r_prdn = metrics::prdn(&fx.orig, &fx.recon);
        check(&mut divergences, &fx.name, "prdn", r_prdn, fx.py_prdn.0, TOL);

        let r_pearson = metrics::pearson_r(&fx.orig, &fx.recon);
        check(
            &mut divergences,
            &fx.name,
            "pearson_r",
            r_pearson,
            fx.py_pearson_r.0,
            TOL,
        );

        let r_snr = metrics::snr_db(&fx.orig, &fx.recon);
        check(&mut divergences, &fx.name, "snr_db", r_snr, fx.py_snr_db.0, TOL);

        // Cross-check: the Rust prd/pearson must ALSO be within tolerance of
        // the SECOND Python implementation (lamquant_codec.lqs). This guards
        // against drift in either Python source, not just the canonical one.
        check(
            &mut divergences,
            &fx.name,
            "prd (vs lqs.prd cross-check)",
            r_prd,
            fx.py_prd_lqs.0,
            CROSS_TOL,
        );
        check(
            &mut divergences,
            &fx.name,
            "pearson_r (vs lqs.pearson_r cross-check)",
            r_pearson,
            fx.py_pearson_r_lqs.0,
            CROSS_TOL,
        );

        // Entropy — only on fixtures carrying a histogram.
        if let (Some(counts), Some(py_h)) = (&fx.counts, &fx.py_entropy_bits) {
            let r_h = metrics::entropy_from_counts(counts);
            check(
                &mut divergences,
                &fx.name,
                "entropy_from_counts",
                r_h,
                py_h.0,
                TOL,
            );
        }

        // Compression ratio — only on fixtures carrying byte counts.
        if let (Some(raw), Some(comp), Some(py_cr)) =
            (fx.raw_bytes, fx.comp_bytes, &fx.py_cr)
        {
            let r_cr = metrics::compression_ratio(raw, comp);
            check(
                &mut divergences,
                &fx.name,
                "compression_ratio",
                r_cr,
                py_cr.0,
                TOL,
            );
        }
    }

    if !divergences.is_empty() {
        let mut msg = format!(
            "\nNUMERICAL PARITY FAILURE: {} metric(s) diverged from the Python golden \
             beyond tolerance (abs/rel {TOL:e}). Rust has NOT reached parity on these — \
             do NOT loosen the tolerance to hide them:\n",
            divergences.len()
        );
        for d in &divergences {
            msg.push_str(&format!(
                "  [{}] {}: rust={:.17e}  python={:.17e}  |Δ|={:.3e}\n",
                d.fixture,
                d.metric,
                d.rust,
                d.python,
                (d.rust - d.python).abs(),
            ));
        }
        panic!("{msg}");
    }
}
