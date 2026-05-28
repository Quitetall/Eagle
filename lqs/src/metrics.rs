//! Canonical LQS metric formulas — one definition each.
//!
//! Ported from `ai_models/metrics.py` (the training-pipeline source of
//! truth) and `lamquant_codec/lqs.py`. The two co-equal primary metrics
//! are R (Pearson correlation, shape preservation) and PRD (percentage
//! root-mean-square difference, magnitude preservation).
//!
//! Float vs integer PRD: the lossy tiers use [`prd`] over `f64` samples.
//! The L (lossless) tier instead asks [`prd_is_exact_zero`] over the
//! INTEGER sample domain — float-converting integer samples and taking a
//! PRD would yield ~1e-12 roundoff and spuriously fail the exact-zero
//! lossless gate.

/// Percentage Root-mean-square Difference (float domain), for lossy tiers.
///
/// `PRD = 100 * sqrt(sum((x - x_hat)^2) / sum(x^2))`
///
/// All-zero guard (matches `prd_numpy`): if `sum(x^2)` is below `eps` the
/// result is `0.0` when the residual is also ~0, else `100.0`.
/// A length mismatch is handled by truncating to the shorter slice.
pub fn prd(orig: &[f64], recon: &[f64]) -> f64 {
    const EPS: f64 = 1e-12;
    let n = orig.len().min(recon.len());
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for i in 0..n {
        let d = orig[i] - recon[i];
        num += d * d;
        den += orig[i] * orig[i];
    }
    if den < EPS {
        return if num < EPS { 0.0 } else { 100.0 };
    }
    100.0 * (num / den).sqrt()
}

/// Bit-exact lossless check on the INTEGER sample domain.
///
/// Returns `true` iff the two integer sample streams are element-wise
/// equal (same length and same values). This is the canonical L-tier
/// PRD gate: exact integer equality == PRD of exactly zero, with no
/// float roundoff. A length mismatch is an automatic `false`.
pub fn prd_is_exact_zero(orig: &[i64], recon: &[i64]) -> bool {
    orig.len() == recon.len() && orig == recon
}

/// Normalized (mean-subtracted) PRD.
///
/// `PRDN = 100 * sqrt(sum((x - x_hat)^2) / sum((x - mean(x))^2))`
///
/// Removes the DC component from the denominator so a large signal
/// offset doesn't deflate the reported error. Same all-zero guard as
/// [`prd`], applied to the mean-subtracted energy.
pub fn prdn(orig: &[f64], recon: &[f64]) -> f64 {
    const EPS: f64 = 1e-12;
    let n = orig.len().min(recon.len());
    if n == 0 {
        return 0.0;
    }
    let mean = orig[..n].iter().sum::<f64>() / n as f64;
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for i in 0..n {
        let d = orig[i] - recon[i];
        num += d * d;
        let c = orig[i] - mean;
        den += c * c;
    }
    if den < EPS {
        return if num < EPS { 0.0 } else { 100.0 };
    }
    100.0 * (num / den).sqrt()
}

/// Pearson correlation coefficient (flatten + truncate + std guard).
///
/// Both slices are treated as flat vectors and truncated to the shorter
/// length. If either input's standard deviation is below `1e-8` the
/// correlation is undefined; we return `1.0` when the two streams are
/// element-wise (approximately) equal and `0.0` otherwise — matching the
/// Python `pearson_r` guard.
pub fn pearson_r(orig: &[f64], recon: &[f64]) -> f64 {
    const STD_EPS: f64 = 1e-8;
    let n = orig.len().min(recon.len());
    if n == 0 {
        return 0.0;
    }
    let nf = n as f64;
    let mx = orig[..n].iter().sum::<f64>() / nf;
    let my = recon[..n].iter().sum::<f64>() / nf;

    let mut sxx = 0.0f64;
    let mut syy = 0.0f64;
    let mut sxy = 0.0f64;
    for i in 0..n {
        let dx = orig[i] - mx;
        let dy = recon[i] - my;
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }

    // Population std (divide by n) — only used to decide the degenerate
    // branch, so the n vs n-1 choice does not affect the returned r.
    let std_x = (sxx / nf).sqrt();
    let std_y = (syy / nf).sqrt();
    if std_x < STD_EPS || std_y < STD_EPS {
        let equal = orig[..n]
            .iter()
            .zip(&recon[..n])
            .all(|(a, b)| (a - b).abs() <= 1e-8);
        return if equal { 1.0 } else { 0.0 };
    }

    let den = (sxx * syy).sqrt();
    if den == 0.0 {
        return 0.0;
    }
    sxy / den
}

/// Signal-to-noise ratio in dB.
///
/// `SNR = 10 * log10(mean(x^2) / mean((x - x_hat)^2))`
///
/// When the noise power is below `1e-30` (effectively perfect recon) the
/// result is capped at `120.0` dB, matching the Python `snr_db`.
pub fn snr_db(orig: &[f64], recon: &[f64]) -> f64 {
    const NOISE_EPS: f64 = 1e-30;
    let n = orig.len().min(recon.len());
    if n == 0 {
        return 120.0;
    }
    let nf = n as f64;
    let mut sig = 0.0f64;
    let mut noise = 0.0f64;
    for i in 0..n {
        sig += orig[i] * orig[i];
        let d = orig[i] - recon[i];
        noise += d * d;
    }
    let sig = sig / nf;
    let noise = noise / nf;
    if noise < NOISE_EPS {
        return 120.0;
    }
    10.0 * (sig / noise).log10()
}

/// Compression ratio for one window: `raw_bytes / max(comp_bytes, 1)`.
pub fn compression_ratio(raw_bytes: u64, comp_bytes: u64) -> f64 {
    raw_bytes as f64 / comp_bytes.max(1) as f64
}

/// Aggregate compression ratio across windows.
///
/// Pooled ratio = `sum(raw) / max(sum(comp), 1)`, which weights each
/// window by its byte size — the correct way to combine ratios (a plain
/// mean of per-window ratios would over-weight tiny windows).
pub fn aggregate_cr(pairs: &[(u64, u64)]) -> f64 {
    let mut raw = 0u64;
    let mut comp = 0u64;
    for (r, c) in pairs {
        raw += *r;
        comp += *c;
    }
    raw as f64 / comp.max(1) as f64
}

/// Quality score: `CR / PRD`, guarded for `PRD <= 0`.
///
/// Higher is better (more compression per unit of distortion). When PRD
/// is zero or negative (lossless / degenerate) the score is the raw CR,
/// treating distortion as a floor of 1.0 so a perfect codec still ranks
/// by its compression ratio rather than blowing up to infinity.
pub fn qs(cr: f64, prd: f64) -> f64 {
    if prd <= 0.0 {
        cr
    } else {
        cr / prd
    }
}

/// Shannon entropy in bits from a histogram of symbol counts.
///
/// `H = -sum_i p_i * log2(p_i)` where `p_i = count_i / total`.
/// Zero-count bins contribute nothing. An empty or all-zero histogram
/// has entropy `0.0`.
pub fn entropy_from_counts(counts: &[u64]) -> f64 {
    let total: u64 = counts.iter().sum();
    if total == 0 {
        return 0.0;
    }
    let total_f = total as f64;
    let mut h = 0.0f64;
    for &c in counts {
        if c == 0 {
            continue;
        }
        let p = c as f64 / total_f;
        h -= p * p.log2();
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prd_of_identical_is_zero() {
        let x = vec![1.0, 2.0, 3.0, 4.0, -5.0];
        assert_eq!(prd(&x, &x), 0.0);
    }

    #[test]
    fn prd_all_zero_guard() {
        let z = vec![0.0, 0.0, 0.0];
        // Identical zeros: 0.0.
        assert_eq!(prd(&z, &z), 0.0);
        // Zero original but nonzero recon: 100.0.
        assert_eq!(prd(&z, &[0.0, 1.0, 0.0]), 100.0);
    }

    #[test]
    fn prd_known_vector() {
        // x = [3,4], xhat = [0,0]: num = 9+16 = 25, den = 9+16 = 25,
        // PRD = 100 * sqrt(1) = 100.
        let x = vec![3.0, 4.0];
        let xhat = vec![0.0, 0.0];
        assert!((prd(&x, &xhat) - 100.0).abs() < 1e-9);

        // x = [10, 0], xhat = [9, 0]: num = 1, den = 100, PRD = 10.
        let a = vec![10.0, 0.0];
        let b = vec![9.0, 0.0];
        assert!((prd(&a, &b) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn exact_zero_integer_domain() {
        let x = vec![1i64, -2, 3, 1000, -32768];
        let y = x.clone();
        assert!(prd_is_exact_zero(&x, &y));

        let mut z = x.clone();
        z[2] += 1; // off by one LSB
        assert!(!prd_is_exact_zero(&x, &z));

        // Length mismatch is not exact.
        assert!(!prd_is_exact_zero(&x, &x[..4]));
    }

    #[test]
    fn pearson_of_identical_is_one() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((pearson_r(&x, &x) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn pearson_perfect_anticorrelation() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![4.0, 3.0, 2.0, 1.0];
        assert!((pearson_r(&x, &y) + 1.0).abs() < 1e-9);
    }

    #[test]
    fn pearson_known_linear() {
        // y = 2x + 1 is a perfect positive linear relationship: r = 1.
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y: Vec<f64> = x.iter().map(|v| 2.0 * v + 1.0).collect();
        assert!((pearson_r(&x, &y) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn pearson_flat_guard() {
        // Both flat & equal => 1.0; flat but unequal => 0.0.
        let a = vec![5.0, 5.0, 5.0];
        assert_eq!(pearson_r(&a, &a), 1.0);
        let b = vec![7.0, 7.0, 7.0];
        assert_eq!(pearson_r(&a, &b), 0.0);
    }

    #[test]
    fn prdn_known_vector() {
        // x = [0, 2], mean = 1, den = (0-1)^2 + (2-1)^2 = 2.
        // recon = [0, 0]: num = 0 + 4 = 4. PRDN = 100*sqrt(4/2) = 141.42...
        let x = vec![0.0, 2.0];
        let r = vec![0.0, 0.0];
        let expected = 100.0 * (4.0f64 / 2.0).sqrt();
        assert!((prdn(&x, &r) - expected).abs() < 1e-9);
    }

    #[test]
    fn snr_identical_capped() {
        let x = vec![1.0, 2.0, 3.0];
        assert_eq!(snr_db(&x, &x), 120.0);
    }

    #[test]
    fn snr_known() {
        // sig power mean = mean([100]) over [10] = 100. noise: recon [9]
        // => mean([1]) = 1. SNR = 10*log10(100/1) = 20 dB.
        let x = vec![10.0];
        let y = vec![9.0];
        assert!((snr_db(&x, &y) - 20.0).abs() < 1e-9);
    }

    #[test]
    fn cr_basics() {
        assert_eq!(compression_ratio(1000, 100), 10.0);
        // comp clamped to 1 to avoid div-by-zero.
        assert_eq!(compression_ratio(50, 0), 50.0);
    }

    #[test]
    fn aggregate_cr_pools_bytes() {
        // (1000/100) and (1000/900) do NOT average to the pooled ratio;
        // pooled = 2000 / 1000 = 2.0.
        let pairs = vec![(1000u64, 100u64), (1000u64, 900u64)];
        assert!((aggregate_cr(&pairs) - 2.0).abs() < 1e-12);
        assert_eq!(aggregate_cr(&[]), 0.0);
    }

    #[test]
    fn qs_guarded() {
        assert_eq!(qs(40.0, 8.0), 5.0);
        // PRD <= 0 returns the raw CR.
        assert_eq!(qs(40.0, 0.0), 40.0);
        assert_eq!(qs(40.0, -1.0), 40.0);
    }

    #[test]
    fn entropy_uniform_two_symbols() {
        // Two equally-likely symbols => 1 bit.
        assert!((entropy_from_counts(&[5, 5]) - 1.0).abs() < 1e-12);
        // Four equally-likely => 2 bits.
        assert!((entropy_from_counts(&[3, 3, 3, 3]) - 2.0).abs() < 1e-12);
        // A single symbol => 0 bits.
        assert_eq!(entropy_from_counts(&[10]), 0.0);
        // Empty / all-zero => 0.
        assert_eq!(entropy_from_counts(&[]), 0.0);
        assert_eq!(entropy_from_counts(&[0, 0]), 0.0);
    }
}
