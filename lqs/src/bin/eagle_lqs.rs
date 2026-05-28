//! `eagle-lqs` — CLI front-end for the LQS vendor-neutral EEG codec
//! benchmark standard.
//!
//! Runs the LQS harness over a built-in synthetic multichannel signal so
//! the binary is useful with zero external data, prints the human report
//! table, and exits non-zero if the codec is below the alerting floor.
//!
//! Usage:
//!
//! ```text
//! eagle-lqs [CODEC]
//! ```
//!
//! `CODEC` is one of:
//!   - `store`    identity passthrough (lossless baseline, default)
//!   - `gzip`     pure-Rust gzip (lossless)
//!   - `quantize` a deliberately-lossy demo codec (÷8 then ×8)
//!
//! Real-EDF corpus loading is a TODO — for now the CLI always grades the
//! built-in fixture so it runs anywhere.

use std::process::ExitCode;

use lqs::adapter::{serialize, Codec, Gzip, Store};
use lqs::harness;

/// A deliberately-lossy demo codec: quantize each sample by integer
/// division by `STEP`, store the quantized values losslessly, and on
/// decode multiply back by `STEP`. The low-order bits are discarded, so
/// reconstruction is NOT bit-exact — exactly the kind of codec the LQS
/// lossy battery is meant to grade. Lives in the CLI as a demonstration
/// (the library ships only lossless reference adapters).
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

    fn encode(&self, signal: &[Vec<i64>], _fs: f64) -> Vec<u8> {
        let q: Vec<Vec<i64>> = signal
            .iter()
            .map(|chan| chan.iter().map(|&s| s / self.step).collect())
            .collect();
        serialize(&q)
    }

    fn decode(&self, blob: &[u8]) -> Vec<Vec<i64>> {
        let q = lqs::adapter::deserialize(blob);
        q.into_iter()
            .map(|chan| chan.into_iter().map(|s| s * self.step).collect())
            .collect()
    }
}

/// Build a synthetic multichannel EEG-like signal: a handful of channels,
/// each a sum of sinusoids placed in distinct clinical bands plus a small
/// DC term, rounded to integer ADC counts. Deterministic — no RNG — so
/// the CLI output is reproducible.
fn synthetic_signal(n_chan: usize, n: usize, fs: f64) -> Vec<Vec<i64>> {
    use std::f64::consts::PI;
    (0..n_chan)
        .map(|c| {
            let amp = 1.0 + 0.3 * c as f64; // per-channel gain
            (0..n)
                .map(|i| {
                    let t = i as f64 / fs;
                    let v = amp
                        * (40.0                                // DC -> sub-delta
                            + 120.0 * (2.0 * PI * 2.0 * t).sin()   // 2 Hz  -> delta
                            + 80.0 * (2.0 * PI * 6.0 * t).sin()    // 6 Hz  -> theta
                            + 60.0 * (2.0 * PI * 10.0 * t).sin()   // 10 Hz -> alpha
                            + 30.0 * (2.0 * PI * 20.0 * t).sin()   // 20 Hz -> beta
                            + 15.0 * (2.0 * PI * 40.0 * t).sin()); // 40 Hz -> gamma
                    v.round() as i64
                })
                .collect()
        })
        .collect()
}

fn main() -> ExitCode {
    let codec_name = std::env::args().nth(1).unwrap_or_else(|| "store".to_string());

    let fs = 256.0;
    let signal = synthetic_signal(4, 512, fs);

    println!("LQS — vendor-neutral EEG codec benchmark standard");
    println!(
        "Fixture: {} channels x {} samples @ {} Hz (synthetic)\n",
        signal.len(),
        signal.first().map(|c| c.len()).unwrap_or(0),
        fs,
    );

    let codec: Box<dyn Codec> = match codec_name.as_str() {
        "store" => Box::new(Store),
        "gzip" => Box::new(Gzip),
        "quantize" => Box::new(Quantize { step: 8 }),
        other => {
            eprintln!("unknown codec '{other}'; valid: store | gzip | quantize");
            return ExitCode::from(2);
        }
    };

    let report = harness::run(codec.as_ref(), &signal, fs);

    print!("{}", report.human_table());
    println!("\n{}", report_badge(&report));

    if report.passed() {
        ExitCode::SUCCESS
    } else {
        // Below the alerting floor: signal failure to the shell.
        ExitCode::FAILURE
    }
}

/// One-line LQS badge derived from a full report's grade.
fn report_badge(report: &lqs::report::LqsReport) -> String {
    if report.passed() {
        format!("LQS-{} COMPLIANT", report.grade)
    } else {
        "LQS NON-COMPLIANT (below alerting floor)".to_string()
    }
}
