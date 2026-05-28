//! LQS tier spec and the grading gate.
//!
//! This module is the authoritative source for the LQS thresholds and
//! the `grade` logic. The tier thresholds and per-band requirement
//! tables are ported verbatim from the Python reference
//! (`lamquant_codec/lqs.py`), with the L-tier PRD gate redefined as an
//! EXACT-ZERO short-circuit (see [`grade`]).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Per-frequency-band quality requirement.
///
/// `freq_range` is kept for documentation / reporting parity with the
/// Python spec but the gate only consults `max_prd` / `min_r`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BandRequirement {
    /// Inclusive-low, exclusive-high band edges in Hz (informational).
    pub freq_range: (f64, f64),
    /// Maximum allowed per-band PRD (percent). Lower is better.
    pub max_prd: f64,
    /// Minimum allowed per-band Pearson R. Higher is better.
    pub min_r: f64,
}

impl BandRequirement {
    /// Convenience constructor matching the Python `BandRequirement(...)`.
    pub fn new(lo: f64, hi: f64, max_prd: f64, min_r: f64) -> Self {
        Self {
            freq_range: (lo, hi),
            max_prd,
            min_r,
        }
    }
}

/// One tier of the LQS standard.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LqsLevel {
    /// Human-readable tier name, e.g. "Clinical".
    pub name: String,
    /// Single-character tier code: 'L', 'C', 'M', or 'A'.
    pub level: char,
    /// Maximum allowed global PRD (percent). For the L tier this field
    /// is documentary only — the L gate is an exact-zero short-circuit.
    pub max_prd: f64,
    /// Minimum allowed global Pearson R.
    pub min_r: f64,
    /// Maximum allowed SNR loss (dB). Reported, not gated, in this port.
    pub max_snr_loss: f64,
    /// Minimum required compression ratio (raw / compressed).
    pub min_cr: f64,
    /// Per-band fidelity requirements keyed by band name.
    pub band_fidelity: BTreeMap<String, BandRequirement>,
}

/// The result of grading a set of metrics against the LQS standard.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ComplianceResult {
    /// Highest passing tier code, or '\0' if below the alerting floor.
    pub grade: char,
    /// Why the next-higher tier failed — the to-do list to climb a tier.
    /// Empty when the codec already passes the strictest tier (L or C).
    pub violations: Vec<String>,
}

impl ComplianceResult {
    /// True iff the codec reached any compliant tier (grade != '\0').
    pub fn passed(&self) -> bool {
        self.grade != '\0'
    }

    /// The grade as a string, or "" for the below-floor sentinel.
    pub fn grade_str(&self) -> String {
        if self.grade == '\0' {
            String::new()
        } else {
            self.grade.to_string()
        }
    }
}

/// Build the canonical LQS tier table.
///
/// Returned in strictness order L, C, M, A. Thresholds and band tables
/// are ported verbatim from `lamquant_codec/lqs.py`, except the L tier
/// whose PRD gate is an exact-zero short-circuit handled in [`grade`]
/// and whose `min_cr` is locked to 0.8 by the vendor-neutral spec.
pub fn levels() -> Vec<LqsLevel> {
    let mut out = Vec::with_capacity(4);

    // ── L : Lossless ────────────────────────────────────────────────
    // No band requirements; PRD gate is special-cased EXACT-ZERO.
    out.push(LqsLevel {
        name: "Lossless".to_string(),
        level: 'L',
        max_prd: 0.0,
        min_r: 1.0,
        max_snr_loss: 0.0,
        min_cr: 0.8,
        band_fidelity: BTreeMap::new(),
    });

    // ── C : Clinical ────────────────────────────────────────────────
    {
        let mut bands = BTreeMap::new();
        bands.insert("delta".to_string(), BandRequirement::new(0.5, 4.0, 5.0, 0.98));
        bands.insert("theta".to_string(), BandRequirement::new(4.0, 8.0, 7.0, 0.97));
        bands.insert("alpha".to_string(), BandRequirement::new(8.0, 13.0, 8.0, 0.96));
        bands.insert("beta".to_string(), BandRequirement::new(13.0, 30.0, 12.0, 0.93));
        bands.insert("gamma".to_string(), BandRequirement::new(30.0, 50.0, 20.0, 0.85));
        out.push(LqsLevel {
            name: "Clinical".to_string(),
            level: 'C',
            max_prd: 9.0,
            min_r: 0.95,
            max_snr_loss: 3.0,
            min_cr: 20.0,
            band_fidelity: bands,
        });
    }

    // ── M : Monitoring ──────────────────────────────────────────────
    {
        let mut bands = BTreeMap::new();
        bands.insert("delta".to_string(), BandRequirement::new(0.5, 4.0, 10.0, 0.95));
        bands.insert("theta".to_string(), BandRequirement::new(4.0, 8.0, 12.0, 0.93));
        bands.insert("alpha".to_string(), BandRequirement::new(8.0, 13.0, 15.0, 0.90));
        bands.insert("beta".to_string(), BandRequirement::new(13.0, 30.0, 25.0, 0.80));
        bands.insert("gamma".to_string(), BandRequirement::new(30.0, 50.0, 40.0, 0.60));
        out.push(LqsLevel {
            name: "Monitoring".to_string(),
            level: 'M',
            max_prd: 20.0,
            min_r: 0.85,
            max_snr_loss: 6.0,
            min_cr: 100.0,
            band_fidelity: bands,
        });
    }

    // ── A : Alerting ────────────────────────────────────────────────
    {
        let mut bands = BTreeMap::new();
        bands.insert("delta".to_string(), BandRequirement::new(0.5, 4.0, 20.0, 0.85));
        bands.insert("theta".to_string(), BandRequirement::new(4.0, 8.0, 25.0, 0.80));
        bands.insert("alpha".to_string(), BandRequirement::new(8.0, 13.0, 30.0, 0.75));
        bands.insert("beta".to_string(), BandRequirement::new(13.0, 30.0, 40.0, 0.65));
        bands.insert("gamma".to_string(), BandRequirement::new(30.0, 50.0, 60.0, 0.40));
        out.push(LqsLevel {
            name: "Alerting".to_string(),
            level: 'A',
            max_prd: 40.0,
            min_r: 0.70,
            max_snr_loss: 10.0,
            min_cr: 200.0,
            band_fidelity: bands,
        });
    }

    out
}

/// Look up a single tier by its character code.
pub fn level_by_char(c: char) -> Option<LqsLevel> {
    levels().into_iter().find(|l| l.level == c)
}

/// Check one lossy tier (C / M / A) against the supplied metrics.
///
/// Returns the list of violations; an empty list means the tier passes.
/// `per_band` is a slice of `(band_name, band_r, band_prd)` triples.
/// A band requirement only constrains the metrics if a matching band
/// name is present in `per_band`; unmeasured bands are not penalized
/// (mirrors the Python `_check_level`, which skips `None` band values).
fn check_lossy(
    level: &LqsLevel,
    r: f64,
    prd: f64,
    cr: f64,
    per_band: &[(String, f64, f64)],
) -> Vec<String> {
    let mut v = Vec::new();

    if r < level.min_r {
        v.push(format!("global R {r:.4} < {:.4}", level.min_r));
    }
    if prd > level.max_prd {
        v.push(format!("global PRD {prd:.2}% > {:.2}%", level.max_prd));
    }
    if cr < level.min_cr {
        v.push(format!("CR {cr:.1} < {:.1}", level.min_cr));
    }

    for (band_name, req) in &level.band_fidelity {
        if let Some((_, br, bp)) = per_band.iter().find(|(n, _, _)| n == band_name) {
            if *br < req.min_r {
                v.push(format!("{band_name} R {br:.4} < {:.4}", req.min_r));
            }
            if *bp > req.max_prd {
                v.push(format!("{band_name} PRD {bp:.2}% > {:.2}%", req.max_prd));
            }
        }
    }

    v
}

/// Grade a set of metrics against the LQS standard.
///
/// Returns the highest passing tier and, when relevant, the violations
/// that blocked the next-higher tier (the climb-a-tier to-do list).
///
/// Gate order:
///
/// 1. **LQS-L short-circuit.** If `prd == 0.0` exactly and `cr >= 0.8`,
///    grade `'L'` and stop. PRD of exactly zero on the integer sample
///    domain means bit-exact reconstruction, which is the lossless
///    contract. (Callers should derive this `prd` via
///    [`crate::metrics::prd_is_exact_zero`] on integer samples, not via
///    the float PRD, to avoid ~1e-12 roundoff spuriously failing the
///    exact-zero test.)
/// 2. **C → M → A descent.** A lossy tier passes iff the global R, PRD,
///    and CR thresholds are met AND every measured band meets its
///    per-band R and PRD floors. The highest fully-passing tier wins.
/// 3. **Below floor.** If no tier passes, grade is the `'\0'` sentinel.
///
/// `per_band` is a slice of `(band_name, band_r, band_prd)` triples.
/// Pass an empty slice to gate on the global metrics only.
pub fn grade(
    r: f64,
    prd: f64,
    cr: f64,
    _snr_loss: f64,
    per_band: &[(String, f64, f64)],
) -> ComplianceResult {
    let tiers = levels();

    // 1. LQS-L exact-zero short-circuit.
    if prd == 0.0 && cr >= 0.8 {
        return ComplianceResult {
            grade: 'L',
            violations: Vec::new(),
        };
    }

    // 2. Descend C -> M -> A. Remember the violations that blocked the
    //    strictest tier we tried, so when we settle on (say) M, the
    //    reported violations explain why C failed.
    let lossy_order = ['C', 'M', 'A'];
    let mut blocking: Vec<String> = Vec::new();
    let mut have_blocking = false;

    for code in lossy_order {
        let level = tiers
            .iter()
            .find(|l| l.level == code)
            .expect("lossy tier present in table");
        let violations = check_lossy(level, r, prd, cr, per_band);
        if violations.is_empty() {
            return ComplianceResult {
                grade: code,
                violations: blocking,
            };
        }
        if !have_blocking {
            blocking = violations;
            have_blocking = true;
        }
    }

    // 3. Below the alerting floor.
    ComplianceResult {
        grade: '\0',
        violations: blocking,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn good_clinical_bands() -> Vec<(String, f64, f64)> {
        // (name, R, PRD) comfortably inside the C-tier floors.
        vec![
            ("delta".to_string(), 0.99, 3.0),
            ("theta".to_string(), 0.98, 4.0),
            ("alpha".to_string(), 0.97, 5.0),
            ("beta".to_string(), 0.95, 8.0),
            ("gamma".to_string(), 0.90, 12.0),
        ]
    }

    #[test]
    fn table_has_four_tiers_in_order() {
        let t = levels();
        assert_eq!(t.len(), 4);
        assert_eq!(t[0].level, 'L');
        assert_eq!(t[1].level, 'C');
        assert_eq!(t[2].level, 'M');
        assert_eq!(t[3].level, 'A');
        // L tier carries the vendor-neutral CR floor.
        assert_eq!(t[0].min_cr, 0.8);
        assert!(t[0].band_fidelity.is_empty());
    }

    #[test]
    fn lossless_short_circuit() {
        // prd == 0 exactly and cr >= 0.8 => 'L'.
        let res = grade(1.0, 0.0, 0.8, 0.0, &[]);
        assert_eq!(res.grade, 'L');
        assert!(res.violations.is_empty());
        assert!(res.passed());

        // Even with garbage R it is still L: exact-zero is bit-exact.
        let res2 = grade(0.0, 0.0, 5.0, 0.0, &[]);
        assert_eq!(res2.grade, 'L');
    }

    #[test]
    fn lossless_blocked_by_cr_floor() {
        // prd == 0 but cr below 0.8 cannot be L; falls through. With
        // bad R it lands below the floor.
        let res = grade(0.0, 0.0, 0.5, 0.0, &[]);
        assert_ne!(res.grade, 'L');
    }

    #[test]
    fn clinical_pass() {
        // prd=5, r=0.96, cr=25 + good bands => 'C'.
        let res = grade(0.96, 5.0, 25.0, 0.0, &good_clinical_bands());
        assert_eq!(res.grade, 'C');
        assert!(res.violations.is_empty());
    }

    #[test]
    fn clinical_pass_global_only() {
        // No band info supplied: gate on globals only, still 'C'.
        let res = grade(0.96, 5.0, 25.0, 0.0, &[]);
        assert_eq!(res.grade, 'C');
    }

    #[test]
    fn below_floor() {
        // prd=50, r=0.5 => '' (below alerting floor).
        let res = grade(0.5, 50.0, 1.0, 0.0, &[]);
        assert_eq!(res.grade, '\0');
        assert!(!res.passed());
        assert_eq!(res.grade_str(), "");
        // The violations explain why C (the strictest lossy tier) failed.
        assert!(!res.violations.is_empty());
    }

    #[test]
    fn descends_to_monitoring_with_clinical_todo() {
        // Passes M globals (r>=0.85, prd<=20, cr>=100) but fails C
        // (cr<20? no — set cr=120; fail C on prd>9 and r<0.95).
        let res = grade(0.90, 15.0, 120.0, 0.0, &[]);
        assert_eq!(res.grade, 'M');
        // Reported violations are the C-tier to-do list.
        assert!(res.violations.iter().any(|s| s.contains("PRD")));
        assert!(res.violations.iter().any(|s| s.contains('R')));
    }

    #[test]
    fn band_failure_drops_tier() {
        // Globals pass C, but a band R is below the C floor => C fails,
        // and M floors (looser) are met, so we land on M.
        let mut bands = good_clinical_bands();
        // Tank gamma R below C's 0.85 but keep it above M's 0.60.
        bands[4] = ("gamma".to_string(), 0.70, 12.0);
        let res = grade(0.96, 5.0, 120.0, 0.0, &bands);
        assert_eq!(res.grade, 'M');
        assert!(res.violations.iter().any(|s| s.contains("gamma")));
    }
}
