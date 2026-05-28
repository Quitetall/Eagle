//! Per-EEG-band fidelity helpers.
//!
//! The Rust analogue of `ai_models/metrics.per_band_prd`: split a signal
//! into the canonical clinical EEG bands and measure per-band fidelity by
//! calling the one-definition metric formulas in [`crate::metrics`].
//!
//! ## Method — FFT-bin masking, std-only
//!
//! Band-splitting here is done by a real-input DFT, a rectangular
//! frequency-domain mask that keeps only the bins inside the band, and an
//! inverse DFT back to the time domain. The mask is rebuilt once per band
//! and applied to BOTH the original and the reconstruction. This is the
//! load-bearing property of per-band fidelity: identical filtering on both
//! inputs means the measured PRD / R / SNR reflect *codec* error in that
//! band, not a filter mismatch between two differently-filtered signals.
//!
//! The DFT is a hand-rolled O(N^2) transform. For the short per-window EEG
//! segments LQS grades this is fast enough and keeps the crate std-only
//! (no `rustfft` / FFT dependency). If a window length ever makes O(N^2)
//! the bottleneck, swap [`rfft`]/[`irfft`] for a power-of-two FFT behind
//! the same signature — the masking logic above stays unchanged.

use std::f64::consts::PI;

use crate::metrics;

/// The five "legacy" canonical EEG bands as `(name, low_hz, high_hz)`.
///
/// Kept for backward compatibility with callers that enumerated the
/// original band table (and with the per-band requirement names in
/// [`crate::levels`]). New code measuring per-band fidelity should use
/// [`per_band_fidelity`], which is driven by [`CLINICAL_BANDS`].
pub const EEG_BANDS: [(&str, f64, f64); 5] = [
    ("delta", 0.5, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 50.0),
];

/// The canonical clinical EEG bands as `(name, low_hz, high_hz)`, in order.
///
/// Edges are pinned to the LQS clinical definition (inclusive-low,
/// exclusive-high in Hz):
///
/// | band       | range (Hz) |
/// |------------|------------|
/// | sub-delta  | 0 – 1      |
/// | delta      | 1 – 4      |
/// | theta      | 4 – 8      |
/// | alpha      | 8 – 12     |
/// | beta       | 13 – 30    |
/// | gamma      | 30 – 100   |
///
/// `alpha` uses the 8–12 Hz convention; the 8–13 Hz convention is also
/// in clinical use — change the `alpha` row if your protocol pins 8–13.
/// The names `delta`/`theta`/`alpha`/`beta`/`gamma` deliberately match the
/// per-band requirement keys in [`crate::levels`] so the grading gate can
/// pair a measured band with its tier requirement by name.
pub const CLINICAL_BANDS: [(&str, f64, f64); 6] = [
    ("sub-delta", 0.0, 1.0),
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 12.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 100.0),
];

/// Return the legacy canonical band names in spec order.
pub fn band_names() -> Vec<&'static str> {
    EEG_BANDS.iter().map(|(n, _, _)| *n).collect()
}

/// Return the clinical band names in spec order.
pub fn clinical_band_names() -> Vec<&'static str> {
    CLINICAL_BANDS.iter().map(|(n, _, _)| *n).collect()
}

/// Per-band fidelity: split `orig` and `recon` into the clinical EEG bands
/// and measure each band with the canonical [`crate::metrics`] formulas.
///
/// For each band in [`CLINICAL_BANDS`] the same rectangular frequency mask
/// is applied to both signals (see the module docs — this is what makes
/// the result codec error rather than filter mismatch), then the masked
/// band signals are passed to [`metrics::pearson_r`], [`metrics::prd`] and
/// [`metrics::snr_db`].
///
/// Returns one `(band_name, r, prd, snr)` tuple per band, in spec order.
///
/// # Arguments
/// * `orig`  — the original samples.
/// * `recon` — the reconstructed samples (truncated to the shorter length).
/// * `fs`    — sampling rate in Hz; must be `> 0` for meaningful bins.
pub fn per_band_fidelity(orig: &[f64], recon: &[f64], fs: f64) -> Vec<(String, f64, f64, f64)> {
    let n = orig.len().min(recon.len());

    // Degenerate inputs: no samples, or a non-positive / non-finite rate
    // means no defined frequency bins. Report a neutral row per band
    // (identical-by-construction empty band => r=1, prd=0, snr capped)
    // so downstream grading sees a well-formed table rather than NaNs.
    // `!fs.is_finite() || fs <= 0.0` rejects NaN, 0, and negatives — a
    // bare `fs > 0.0` would let NaN through as "false then unhandled".
    if n == 0 || !fs.is_finite() || fs <= 0.0 {
        return CLINICAL_BANDS
            .iter()
            .map(|(name, _, _)| {
                let empty: [f64; 0] = [];
                (
                    name.to_string(),
                    metrics::pearson_r(&empty, &empty),
                    metrics::prd(&empty, &empty),
                    metrics::snr_db(&empty, &empty),
                )
            })
            .collect();
    }

    let o = &orig[..n];
    let r = &recon[..n];

    // One forward DFT per signal, reused across all band masks.
    let (o_re, o_im) = rfft(o);
    let (r_re, r_im) = rfft(r);

    CLINICAL_BANDS
        .iter()
        .map(|&(name, lo, hi)| {
            let mask = band_mask(n, fs, lo, hi);
            let ob = irfft_masked(&o_re, &o_im, &mask, n);
            let rb = irfft_masked(&r_re, &r_im, &mask, n);
            (
                name.to_string(),
                metrics::pearson_r(&ob, &rb),
                metrics::prd(&ob, &rb),
                metrics::snr_db(&ob, &rb),
            )
        })
        .collect()
}

/// Build a keep/drop mask over the `n/2 + 1` non-negative-frequency DFT
/// bins for the half-open band `[lo, hi)` Hz at sample rate `fs`.
///
/// Bin `k` maps to frequency `k * fs / n`. A bin is kept when its center
/// frequency lands in `[lo, hi)`. The DC bin (`k == 0`, 0 Hz) is included
/// only when `lo <= 0`, so the sub-delta band carries the DC component and
/// the higher bands do not double-count it.
fn band_mask(n: usize, fs: f64, lo: f64, hi: f64) -> Vec<bool> {
    let half = n / 2 + 1;
    let bin_hz = fs / n as f64;
    (0..half)
        .map(|k| {
            let f = k as f64 * bin_hz;
            f >= lo && f < hi
        })
        .collect()
}

/// Naive real-input DFT.
///
/// Returns the non-negative-frequency half-spectrum (`n/2 + 1` complex
/// bins) as parallel real/imag vectors, matching a numpy `rfft`. O(N^2);
/// adequate for the short per-window EEG segments LQS grades and keeps the
/// crate dependency-free.
fn rfft(x: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = x.len();
    let half = n / 2 + 1;
    let mut re = vec![0.0f64; half];
    let mut im = vec![0.0f64; half];
    for (k, (rk, ik)) in re.iter_mut().zip(im.iter_mut()).enumerate() {
        let w = -2.0 * PI * k as f64 / n as f64;
        let mut sr = 0.0f64;
        let mut si = 0.0f64;
        for (j, &xj) in x.iter().enumerate() {
            let ang = w * j as f64;
            sr += xj * ang.cos();
            si += xj * ang.sin();
        }
        *rk = sr;
        *ik = si;
    }
    (re, im)
}

/// Inverse real DFT of a masked half-spectrum back to `n` real samples.
///
/// Reconstructs the full-length real signal from the non-negative half
/// `(re, im)` after applying `mask` (kept bins pass, dropped bins zero).
/// Hermitian symmetry of the negative frequencies is folded in by the
/// `2*cos` term for the interior bins, with DC (and the Nyquist bin when
/// `n` is even) counted once. O(N^2), mirroring [`rfft`].
fn irfft_masked(re: &[f64], im: &[f64], mask: &[bool], n: usize) -> Vec<f64> {
    let half = re.len();
    let even = n.is_multiple_of(2);
    let mut out = vec![0.0f64; n];
    for (j, oj) in out.iter_mut().enumerate() {
        let mut acc = 0.0f64;
        for k in 0..half {
            if !mask[k] {
                continue;
            }
            let ang = 2.0 * PI * k as f64 * j as f64 / n as f64;
            // Bins with a distinct conjugate partner (everything except DC
            // and, for even n, the Nyquist bin) contribute twice via
            // Hermitian symmetry: X[k]*e + conj(X[k])*conj(e) = 2*Re(...).
            let unique = k == 0 || (even && k == half - 1);
            let scale = if unique { 1.0 } else { 2.0 };
            acc += scale * (re[k] * ang.cos() - im[k] * ang.sin());
        }
        *oj = acc / n as f64;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A composite signal with energy spread across several bands.
    fn make_signal(n: usize, fs: f64) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                0.5                                   // DC -> sub-delta
                    + 1.0 * (2.0 * PI * 2.0 * t).sin()  // 2 Hz  -> delta
                    + 0.8 * (2.0 * PI * 10.0 * t).sin() // 10 Hz -> alpha
                    + 0.4 * (2.0 * PI * 40.0 * t).sin() // 40 Hz -> gamma
            })
            .collect()
    }

    #[test]
    fn identical_signals_perfect_per_band() {
        let fs = 256.0;
        let x = make_signal(256, fs);
        let out = per_band_fidelity(&x, &x, fs);
        assert_eq!(out.len(), CLINICAL_BANDS.len());
        for (name, r, prd, snr) in out {
            // Identical inputs => identical band signals => no codec error.
            assert!((r - 1.0).abs() < 1e-9, "{name}: r={r} expected 1");
            assert!(prd.abs() < 1e-9, "{name}: prd={prd} expected 0");
            // snr_db caps perfect recon at 120 dB.
            assert!((snr - 120.0).abs() < 1e-9, "{name}: snr={snr} expected 120");
        }
    }

    #[test]
    fn band_order_and_names_match_spec() {
        let fs = 128.0;
        let x = make_signal(128, fs);
        let out = per_band_fidelity(&x, &x, fs);
        let got: Vec<String> = out.into_iter().map(|(n, ..)| n).collect();
        let want: Vec<String> = CLINICAL_BANDS
            .iter()
            .map(|(n, ..)| n.to_string())
            .collect();
        assert_eq!(got, want);
    }

    #[test]
    fn perturbation_degrades_some_band() {
        // Add a 10 Hz (alpha) error to the recon: the alpha band must show
        // nonzero PRD while a band with no perturbed energy (theta, 4-8 Hz,
        // unoccupied here) stays near-perfect on identical content.
        let fs = 256.0;
        let n = 256;
        let x = make_signal(n, fs);
        let mut y = x.clone();
        for (i, yi) in y.iter_mut().enumerate() {
            let t = i as f64 / fs;
            *yi += 0.2 * (2.0 * PI * 10.0 * t).sin();
        }
        let out = per_band_fidelity(&x, &y, fs);
        let alpha = out.iter().find(|(n, ..)| n == "alpha").unwrap();
        assert!(alpha.2 > 1e-6, "alpha PRD should be nonzero, got {}", alpha.2);
    }

    #[test]
    fn masked_bands_sum_to_original() {
        // FFT-bin masking partitions the spectrum: summing every band's
        // reconstruction must recover the original signal (energy is
        // conserved, no bin is dropped or double-counted across bands).
        let fs = 200.0;
        let n = 200;
        let x = make_signal(n, fs);
        let (re, im) = rfft(&x);
        let mut sum = vec![0.0f64; n];
        for &(_, lo, hi) in CLINICAL_BANDS.iter() {
            let mask = band_mask(n, fs, lo, hi);
            let band = irfft_masked(&re, &im, &mask, n);
            for (s, b) in sum.iter_mut().zip(band.iter()) {
                *s += b;
            }
        }
        // Bands 30-100 cover up to Nyquist (100 Hz); 12-13 Hz is the only
        // intentional gap between alpha (8-12) and beta (13-30). The test
        // signal has no energy there, so the sum reconstructs x exactly.
        for (a, b) in x.iter().zip(sum.iter()) {
            assert!((a - b).abs() < 1e-6, "reconstruction mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn rfft_irfft_roundtrip_full_mask() {
        // Keep-everything mask must invert the DFT back to the input.
        let fs = 64.0;
        let n = 64;
        let x = make_signal(n, fs);
        let (re, im) = rfft(&x);
        let mask = vec![true; re.len()];
        let back = irfft_masked(&re, &im, &mask, n);
        for (a, b) in x.iter().zip(back.iter()) {
            assert!((a - b).abs() < 1e-6, "roundtrip mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn degenerate_inputs_yield_neutral_table() {
        // Empty inputs and a non-positive sample rate both produce a
        // well-formed per-band table (no NaNs), one neutral row per band.
        let empty: Vec<f64> = vec![];
        let out = per_band_fidelity(&empty, &empty, 256.0);
        assert_eq!(out.len(), CLINICAL_BANDS.len());
        for (name, r, prd, snr) in &out {
            assert!(r.is_finite() && prd.is_finite() && snr.is_finite(), "{name} NaN");
        }
        let x = make_signal(16, 256.0);
        let out2 = per_band_fidelity(&x, &x, 0.0);
        assert_eq!(out2.len(), CLINICAL_BANDS.len());
        for (name, r, prd, snr) in &out2 {
            assert!(r.is_finite() && prd.is_finite() && snr.is_finite(), "{name} NaN (fs=0)");
        }
    }
}
