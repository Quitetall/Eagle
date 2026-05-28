//! `eagle` — benchmark the LamQuant lossless codec with the LQS standard.
//!
//! Usage:
//!   eagle [FILE.edf]
//!
//! Resolves the `lml` binary (env LML_BIN → sibling LamQuant-Lossless →
//! PATH), grades it via the vendor-neutral LQS harness, prints the
//! standard report. With no file, grades a built-in synthetic signal.

use eagle::LamQuantLossless;

fn synthetic(channels: usize, samples: usize, fs: f64) -> (Vec<Vec<i64>>, f64) {
    // Deterministic multi-band sinusoid mix, no RNG.
    let mut sig = vec![vec![0i64; samples]; channels];
    for (c, ch) in sig.iter_mut().enumerate() {
        for (n, s) in ch.iter_mut().enumerate() {
            let t = n as f64 / fs;
            let f = 3.0 + 7.0 * (c as f64 + 1.0);
            *s = (8000.0 * (2.0 * std::f64::consts::PI * f * t).sin()) as i64;
        }
    }
    (sig, fs)
}

fn main() {
    let file = std::env::args().nth(1);
    let fs = 256.0;

    let (signal, fs) = match file.as_deref() {
        Some(path) => match lqs::edf::read_edf(path) {
            Ok(edf) => (edf.channels, edf.fs),
            Err(e) => {
                eprintln!("eagle: failed to read EDF {path}: {e}");
                std::process::exit(3);
            }
        },
        None => synthetic(4, 1024, fs),
    };

    let codec = match LamQuantLossless::resolve(fs) {
        Some(c) => c,
        None => {
            eprintln!(
                "eagle: `lml` binary not found. Set LML_BIN, or clone \
                 LamQuant-Lossless as a sibling and build `lml`. \
                 (LQS itself ships neutral reference codecs — try `lqs store`.)"
            );
            std::process::exit(2);
        }
    };

    let report = lqs::harness::run(&codec, &signal, fs);
    println!("{}", report.human_table());
    // Pass = graded into a deployable tier; below-floor → nonzero.
    std::process::exit(if "LCMA".contains(report.grade) { 0 } else { 1 });
}
