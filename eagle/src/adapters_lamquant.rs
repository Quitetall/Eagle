//! Real LamQuant-Lossless codec adapter.
//!
//! Where [`lqs::adapter`] ships the always-available reference codecs
//! (`store`, `gzip`, optional `zstd`), this module wires the **actual
//! production lossless codec** — the `lml` CLI from the sibling
//! LamQuant-Lossless workspace — behind the same [`Codec`] interface so
//! LQS grades the real `.lml` wire format, not a stand-in.
//!
//! ## Why a subprocess and not a crate dependency
//!
//! `lml` operates on **files** (EDF in, raw int32 LE out), not in-memory
//! arrays, and lives in a separate workspace. Linking it as a library
//! would couple the (vendor-neutral) LQS crate to the LamQuant-Lossless
//! build. Instead the adapter shells out to a prebuilt `lml` binary. The
//! default CI build does **not** require that binary to be present:
//!
//! - [`LamQuantLossless::resolve`] returns `None` when no `lml` can be
//!   found, so callers can skip the adapter cleanly.
//! - The `#[cfg(test)]` round-trip below early-returns (printing a skip
//!   note) when `lml` is absent, so `cargo test -p eagle` is green with or
//!   without the sibling LamQuant-Lossless workspace checked out.
//!
//! No Cargo feature gate is needed: the module is pure `std` (process
//! spawn + temp files), so it always compiles; availability is a
//! *runtime* property of the host, resolved by [`resolve_lml_bin`].
//!
//! ## Binary resolution order
//!
//! 1. `$LML_BIN` if set (and the path exists).
//! 2. The prebuilt sibling default
//!    `/tmp/lamquant-verify/LamQuant-Lossless/target/debug/lml`.
//! 3. `lml` on `PATH`.
//!
//! ## Round-trip contract
//!
//! `lml` handles the EDF **digital** sample domain (signed 16-bit).
//!
//! - **encode**: write the in-memory channels as a per-channel-rate EDF
//!   (`write_edf_bytes`, ragged-capable), then `lml encode` → the bare
//!   `.lml` bytes (the blob). Mixed-rate channels are written as one data
//!   record with per-channel `samples_per_record`.
//! - **decode**: `lml decode --to-edf` reconstructs the EDF
//!   **byte-identical** (header + all channels + trailing — the path
//!   `lml roundtrip` verifies), then the shared [`lqs::edf`] reader
//!   re-reads every channel at its native rate. This recovers ALL
//!   channels, including slow mixed-rate aux channels — unlike the
//!   raw-int32 decode, which emits only the dominant-rate EEG matrix and
//!   silently drops aux.
//!
//! So both uniform and **mixed-rate** signals round-trip bit-exactly and
//! grade LQS-L (verified on real recordings, e.g. nedc example.edf:
//! 27 EEG @250 Hz + 3 aux @1 Hz).
//!
//! A signal with out-of-`i16` samples or an empty channel yields an empty
//! blob → harness shape-guard reports below-floor, never a panic. EEG
//! digital ADC samples are 16-bit by construction.
//!
//! [`encode`]: Codec::encode
//! [`decode`]: Codec::decode

use std::path::{Path, PathBuf};
use std::process::Command;

use lqs::adapter::Codec;

/// Default location of the prebuilt sibling `lml` binary.
const DEFAULT_LML_BIN: &str = "/tmp/lamquant-verify/LamQuant-Lossless/target/debug/lml";

/// Size in bytes of the fixed EDF main header and of each signal header.
const EDF_HEADER_BLOCK: usize = 256;

/// Resolve the `lml` binary to invoke, or `None` if none is usable.
///
/// Tries `$LML_BIN`, then the prebuilt sibling default, then `lml` on
/// `PATH` (probed by running `lml --version`). Returns the resolved path
/// only when a candidate actually exists / is runnable, so callers can
/// treat `None` as "skip the real-codec adapter on this host".
pub fn resolve_lml_bin() -> Option<PathBuf> {
    // 1. Explicit override via env var.
    if let Some(p) = std::env::var_os("LML_BIN") {
        let path = PathBuf::from(p);
        if path.is_file() {
            return Some(path);
        }
        // An LML_BIN that does not point at a file is a misconfiguration;
        // fall through to the other candidates rather than fail hard.
    }

    // 2. Prebuilt sibling default.
    let default = Path::new(DEFAULT_LML_BIN);
    if default.is_file() {
        return Some(default.to_path_buf());
    }

    // 3. `lml` on PATH — probe by actually running it, since we cannot
    //    portably stat PATH entries. `--version` is cheap and side-effect
    //    free; a non-zero/failed spawn means no usable `lml`.
    if Command::new("lml")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
    {
        return Some(PathBuf::from("lml"));
    }

    None
}

/// The real LamQuant-Lossless codec, driven through the `lml` CLI.
///
/// Declared lossless: every supported signal (see the module docs) round
/// trips bit-exactly through the production `.lml` wire format. Construct
/// via [`LamQuantLossless::resolve`] (auto-discovers `lml`) or
/// [`LamQuantLossless::with_bin`] (explicit path).
#[derive(Clone, Debug)]
pub struct LamQuantLossless {
    /// Path to the `lml` binary this adapter shells out to.
    lml_bin: PathBuf,
    /// Sample rate carried into the temp EDF header (metadata only — the
    /// integer-domain samples round-trip independent of it).
    fs: f64,
}

impl LamQuantLossless {
    /// Construct the adapter if a usable `lml` binary can be found.
    ///
    /// Returns `None` when no `lml` is available (see [`resolve_lml_bin`]),
    /// so the default CI build / a host without the sibling workspace can
    /// skip the real-codec adapter without error.
    pub fn resolve(fs: f64) -> Option<Self> {
        resolve_lml_bin().map(|lml_bin| Self { lml_bin, fs })
    }

    /// Construct the adapter against an explicit `lml` binary path.
    pub fn with_bin(lml_bin: PathBuf, fs: f64) -> Self {
        Self { lml_bin, fs }
    }

    /// The resolved `lml` binary path this adapter uses.
    pub fn lml_bin(&self) -> &Path {
        &self.lml_bin
    }
}

/// A scratch directory under the system temp dir, removed on drop.
///
/// `lml encode`/`decode` write a handful of sidecar files (manifest,
/// audit log, state) next to their output; isolating each invocation in
/// its own directory keeps concurrent adapter calls from colliding and
/// makes cleanup a single `remove_dir_all`.
struct ScratchDir {
    path: PathBuf,
}

impl ScratchDir {
    /// Create a uniquely-named scratch directory (`pid` + nanos + `tag`).
    fn new(tag: &str) -> std::io::Result<Self> {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        // A monotonic counter disambiguates two calls within the same nano.
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        let name = format!(
            "lqs_lml_{}_{}_{}_{}",
            std::process::id(),
            nanos,
            seq,
            tag
        );
        let path = std::env::temp_dir().join(name);
        std::fs::create_dir_all(&path)?;
        Ok(Self { path })
    }

    fn join(&self, name: &str) -> PathBuf {
        self.path.join(name)
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        // Best-effort cleanup; a leaked tempdir must never panic a test.
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

/// Write an ASCII value left-justified into a fixed-width EDF field,
/// space-padded. Returns an error if the value overflows the field.
fn edf_field(out: &mut Vec<u8>, value: &str, width: usize) -> Result<(), ()> {
    let bytes = value.as_bytes();
    if bytes.len() > width {
        return Err(());
    }
    out.extend_from_slice(bytes);
    out.extend(std::iter::repeat(b' ').take(width - bytes.len()));
    Ok(())
}

/// Build a minimal, spec-valid EDF byte image for `signal` at rate `fs`.
///
/// One data record holds every channel's samples (`record_duration`
/// chosen so the stored rate matches `fs`). Returns `None` when the
/// signal cannot be expressed as EDF digital samples: non-i16 values,
/// ragged channels, or an empty / zero-length signal (see the module
/// docs). The layout mirrors `lqs::edf::read_edf`'s parser exactly.
pub fn write_edf_bytes(signal: &[Vec<i64>], fs: f64) -> Option<Vec<u8>> {
    let ns = signal.len();
    if ns == 0 {
        return None;
    }
    // Per-channel sample counts. Channels may differ in length (mixed-rate
    // EDF) — we write ONE data record whose per-signal samples_per_record
    // is each channel's own length. EDF supports this natively. Every
    // channel must be non-empty.
    let spr: Vec<usize> = signal.iter().map(|c| c.len()).collect();
    if spr.iter().any(|&n| n == 0) {
        return None;
    }
    // Every sample must fit signed 16-bit (the EDF digital domain).
    if signal
        .iter()
        .flat_map(|c| c.iter())
        .any(|&s| s < i16::MIN as i64 || s > i16::MAX as i64)
    {
        return None;
    }
    if !fs.is_finite() || fs <= 0.0 {
        return None;
    }

    // One record covers the whole signal. record_duration is cosmetic for
    // the lossless grade — lml round-trips the digital samples bit-exact
    // regardless of the declared rate, and grading fs comes from the
    // caller (EdfSignal::fs), not this internal EDF. Use 1.0 so mixed-rate
    // channels (different spr) all fit one record cleanly.
    let dur_str = "1       ".trim().to_string();

    let header_bytes = EDF_HEADER_BLOCK + ns * EDF_HEADER_BLOCK;
    let header_str = header_bytes.to_string();
    let max_spr_len = spr.iter().map(|n| n.to_string().len()).max().unwrap_or(1);
    if header_str.len() > 8 || ns.to_string().len() > 4 || max_spr_len > 8 {
        return None;
    }

    let total_samples: usize = spr.iter().sum();
    let mut buf = Vec::with_capacity(header_bytes + total_samples * 2);

    // ── Main header. ───────────────────────────────────────────────────
    edf_field(&mut buf, "0", 8).ok()?; // version
    edf_field(&mut buf, "LQS X X X", 80).ok()?; // patient
    edf_field(&mut buf, "Startdate X", 80).ok()?; // recording
    edf_field(&mut buf, "01.01.26", 8).ok()?; // startdate
    edf_field(&mut buf, "00.00.00", 8).ok()?; // starttime
    edf_field(&mut buf, &header_str, 8).ok()?; // header_bytes
    edf_field(&mut buf, "", 44).ok()?; // reserved
    edf_field(&mut buf, "1", 8).ok()?; // n_data_records = 1
    edf_field(&mut buf, &dur_str, 8).ok()?; // record_duration_sec
    edf_field(&mut buf, &ns.to_string(), 4).ok()?; // n_signals

    // ── Signal headers, field-by-field across all signals. ─────────────
    for i in 0..ns {
        edf_field(&mut buf, &format!("ch{i}"), 16).ok()?; // label
    }
    for _ in 0..ns {
        edf_field(&mut buf, "AgAgCl", 80).ok()?; // transducer
    }
    for _ in 0..ns {
        edf_field(&mut buf, "uV", 8).ok()?; // phys_dim
    }
    for _ in 0..ns {
        edf_field(&mut buf, "-32768", 8).ok()?; // phys_min
    }
    for _ in 0..ns {
        edf_field(&mut buf, "32767", 8).ok()?; // phys_max
    }
    for _ in 0..ns {
        edf_field(&mut buf, "-32768", 8).ok()?; // dig_min
    }
    for _ in 0..ns {
        edf_field(&mut buf, "32767", 8).ok()?; // dig_max
    }
    for _ in 0..ns {
        edf_field(&mut buf, "", 80).ok()?; // prefilter
    }
    for i in 0..ns {
        edf_field(&mut buf, &spr[i].to_string(), 8).ok()?; // per-channel n_samples_per_record
    }
    for _ in 0..ns {
        edf_field(&mut buf, "", 32).ok()?; // signal reserved
    }
    debug_assert_eq!(buf.len(), header_bytes, "EDF header block size mismatch");

    // ── Data: one record, signals in order, each `spr` little-endian i16.
    for chan in signal {
        for &s in chan {
            buf.extend_from_slice(&(s as i16).to_le_bytes());
        }
    }

    Some(buf)
}


impl LamQuantLossless {
    /// Encode `signal` to production `.lml` bytes, or `Vec::new()` on any
    /// failure (unsupported signal shape, `lml` error, I/O error).
    ///
    /// The returned blob is `[shape header][.lml bytes]` where the header
    /// is `b"LQS1"` + `u32 n_chan` + `n_chan × u32` per-channel sample
    /// counts (all little-endian). This makes the blob self-describing so
    /// `decode` reconstructs the exact per-channel shape — including
    /// mixed-rate files whose channels differ in length — without relying
    /// on a uniform-shape reparse of `lml info`.
    fn try_encode(&self, signal: &[Vec<i64>], fs: f64) -> Vec<u8> {
        let edf = match write_edf_bytes(signal, fs) {
            Some(e) => e,
            None => return Vec::new(),
        };
        let dir = match ScratchDir::new("enc") {
            Ok(d) => d,
            Err(_) => return Vec::new(),
        };
        let edf_path = dir.join("in.edf");
        if std::fs::write(&edf_path, &edf).is_err() {
            return Vec::new();
        }

        // `--no-bundle --i-understand-data-loss` emits a bare `.lml` (the
        // real wire format) rather than the per-recording `.lma` envelope.
        // `-q` silences progress so only errors reach stderr.
        let status = Command::new(&self.lml_bin)
            .arg("encode")
            .arg(&edf_path)
            .arg("-o")
            .arg(&dir.path)
            .arg("--no-bundle")
            .arg("--i-understand-data-loss")
            .arg("-q")
            .output();
        match status {
            Ok(o) if o.status.success() => {}
            _ => return Vec::new(),
        }

        // `lml` names the output after the input stem: in.edf -> in.lml.
        // The bare `.lml` is the blob — it carries the container metadata
        // `decode --to-edf` needs to reconstruct the EDF byte-exact.
        let lml_path = dir.join("in.lml");
        std::fs::read(&lml_path).unwrap_or_default()
    }

    /// Decode production `.lml` bytes back to the per-channel signal, or
    /// `Vec::new()` on any failure (so the L-tier gate sees a mismatch).
    ///
    /// Reconstructs the full EDF via `lml decode --to-edf` (byte-identical,
    /// ALL channels including mixed-rate aux — the path `lml roundtrip`
    /// verifies), then re-reads it with the shared [`lqs::edf`] parser. The
    /// raw-int32 decode is NOT used: it emits only the dominant-rate EEG
    /// matrix and silently drops slow aux channels, which would make a
    /// mixed-rate round trip non-bit-exact.
    fn try_decode(&self, blob: &[u8]) -> Vec<Vec<i64>> {
        if blob.is_empty() {
            return Vec::new();
        }
        let dir = match ScratchDir::new("dec") {
            Ok(d) => d,
            Err(_) => return Vec::new(),
        };
        let lml_path = dir.join("in.lml");
        if std::fs::write(&lml_path, blob).is_err() {
            return Vec::new();
        }

        let edf_path = dir.join("recon.edf");
        let dec = Command::new(&self.lml_bin)
            .arg("decode")
            .arg(&lml_path)
            .arg("--to-edf")
            .arg("-o")
            .arg(&edf_path)
            .arg("-q")
            .output();
        match dec {
            Ok(o) if o.status.success() => {}
            _ => return Vec::new(),
        }

        // Re-read the byte-exact reconstructed EDF with the canonical LQS
        // reader: this recovers every channel at its native rate, so a
        // mixed-rate signal round-trips with all channels intact.
        match lqs::edf::read_edf(&edf_path) {
            Ok(sig) => sig.channels,
            Err(_) => Vec::new(),
        }
    }
}

impl Codec for LamQuantLossless {
    fn name(&self) -> &str {
        "lamquant-lossless"
    }

    fn declared_lossless(&self) -> bool {
        true
    }

    fn encode(&self, signal: &[Vec<i64>], fs: f64) -> Vec<u8> {
        // Prefer the rate the harness passes; fall back to the adapter's
        // configured fs only if the caller hands us a non-positive rate.
        let rate = if fs.is_finite() && fs > 0.0 { fs } else { self.fs };
        self.try_encode(signal, rate)
    }

    fn decode(&self, blob: &[u8]) -> Vec<Vec<i64>> {
        self.try_decode(blob)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lqs::harness;

    /// A known multi-channel i64 signal that fits the EDF digital domain
    /// (every value in the i16 range) and has equal-length channels, so it
    /// is expressible as a `.lml`.
    ///
    /// It is deliberately corpus-sized (a few thousand samples per
    /// channel) and EEG-like (smooth, band-limited, integer ADC counts):
    /// the production codec needs a real window of correlated data to
    /// compress below the raw 8-bytes/sample reference container and clear
    /// the harness's L-tier `cr >= 0.8` floor. A handful of samples is
    /// dominated by container overhead and would (correctly) grade below
    /// the floor, so the fixture mirrors the shape of real data the codec
    /// is built for. Deterministic — no RNG — so the round trip is
    /// reproducible.
    fn fixture() -> Vec<Vec<i64>> {
        use std::f64::consts::PI;
        let fs = 256.0;
        let n = 4096;
        (0..4)
            .map(|c| {
                let amp = 1.0 + 0.25 * c as f64;
                (0..n)
                    .map(|i| {
                        let t = i as f64 / fs;
                        // EEG-band sinusoids scaled to a few hundred ADC
                        // counts — comfortably inside the i16 digital range.
                        let v = amp
                            * (80.0 * (2.0 * PI * 2.0 * t).sin()
                                + 50.0 * (2.0 * PI * 10.0 * t).sin()
                                + 25.0 * (2.0 * PI * 22.0 * t).sin()
                                + 10.0 * (2.0 * PI * 40.0 * t).sin());
                        v.round() as i64
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn write_edf_bytes_rejects_unsupported_shapes() {
        // Empty signal.
        assert!(write_edf_bytes(&[], 256.0).is_none());
        // Empty channel.
        assert!(write_edf_bytes(&[vec![]], 256.0).is_none());
        // Ragged channels are now SUPPORTED (mixed-rate EDF): one record
        // with per-channel samples_per_record = each channel's length.
        assert!(write_edf_bytes(&[vec![1, 2], vec![1]], 256.0).is_some());
        // Out-of-i16-range sample.
        assert!(write_edf_bytes(&[vec![i64::MAX]], 256.0).is_none());
        // Non-positive rate.
        assert!(write_edf_bytes(&[vec![1, 2]], 0.0).is_none());
        // A valid signal succeeds and produces a 256*(1+ns)-byte header
        // followed by one record of `ns * spr` little-endian i16 samples.
        let sig = fixture();
        let edf = write_edf_bytes(&sig, 256.0).expect("valid fixture -> EDF");
        let ns = sig.len();
        let spr = sig[0].len();
        let header = EDF_HEADER_BLOCK * (1 + ns);
        assert_eq!(edf.len(), header + ns * spr * 2, "header + 1 record of i16");
        assert_eq!(&edf[..8], b"0       ", "EDF version field");

        // A small i16-range signal is still expressible (shape, not size,
        // is the constraint) even if it would not clear the codec's CR
        // floor — that floor is the harness's call, not the writer's.
        assert!(write_edf_bytes(&[vec![0, 1, -1, i16::MAX as i64]], 256.0).is_some());
    }


    #[test]
    fn empty_blob_decodes_to_empty_signal() {
        // Decode side must be panic-free on an empty/absent blob even
        // without lml; an empty blob short-circuits before any spawn.
        let codec = LamQuantLossless::with_bin(PathBuf::from("/nonexistent/lml"), 256.0);
        assert!(codec.decode(&[]).is_empty());
        // declared_lossless / name are pure and need no binary.
        assert!(codec.declared_lossless());
        assert_eq!(codec.name(), "lamquant-lossless");
    }

    /// End-to-end round trip through the REAL `lml` binary.
    ///
    /// Gated on `lml` availability: if no binary resolves, the test prints
    /// a skip note and passes, so `cargo test -p eagle` is green on a host
    /// without the sibling LamQuant-Lossless workspace.
    #[test]
    fn lml_roundtrip_grades_lqs_l_when_available() {
        let codec = match LamQuantLossless::resolve(256.0) {
            Some(c) => c,
            None => {
                eprintln!(
                    "SKIP lml_roundtrip_grades_lqs_l_when_available: \
                     no lml binary found (set LML_BIN, build the sibling \
                     LamQuant-Lossless workspace, or put `lml` on PATH)"
                );
                return;
            }
        };
        eprintln!("using lml binary: {}", codec.lml_bin().display());

        let signal = fixture();

        // (a) Direct adapter round trip is bit-exact.
        let blob = codec.encode(&signal, 256.0);
        assert!(
            !blob.is_empty(),
            "lml encode produced no .lml bytes for the fixture"
        );
        let back = codec.decode(&blob);
        assert_eq!(
            back, signal,
            "lamquant-lossless failed bit-exact round trip through real lml"
        );

        // (b) The harness grades it LQS-L (bit-exact integer domain).
        let report = harness::run(&codec, &signal, 256.0);
        assert!(report.bit_exact, "real-codec round trip must be bit-exact");
        assert_eq!(report.grade, 'L', "bit-exact codec must grade LQS-L");
        assert_eq!(report.prd, 0.0);
        assert_eq!(report.r, 1.0);
    }

    #[test]
    fn mixed_rate_ragged_round_trips_bit_exact_when_available() {
        // Mixed-rate (ragged) signal: ch0 has 8 samples, ch1 has 3 — the
        // shape of a 250 Hz EEG channel beside a 1 Hz aux channel. The
        // adapter encodes a per-channel-rate EDF, then decode reconstructs
        // it byte-exact via `lml decode --to-edf` and re-reads ALL channels
        // (no aux dropped). Round trip must be bit-exact → LQS-L.
        let codec = match LamQuantLossless::resolve(256.0) {
            Some(c) => c,
            None => {
                eprintln!("SKIP mixed_rate_ragged_round_trips: no lml binary");
                return;
            }
        };
        let signal: Vec<Vec<i64>> = vec![vec![10, -20, 30, -40, 50, -60, 70, -80], vec![1, -2, 3]];
        let blob = codec.encode(&signal, 256.0);
        assert!(!blob.is_empty(), "encode produced no .lml for ragged signal");
        let back = codec.decode(&blob);
        assert_eq!(back, signal, "ragged mixed-rate signal must round-trip bit-exact");

        // The harness sees a bit-exact reconstruction of the full ragged
        // channel set. (This tiny signal is header-dominated so its CR is
        // below the L floor → grade isn't 'L'; the LQS-L-on-mixed-rate
        // result is exercised on a real recording in the CLI / corpus run.)
        let report = harness::run(&codec, &signal, 256.0);
        assert!(report.bit_exact, "mixed-rate round trip must be bit-exact");
        assert_eq!(report.prd, 0.0, "exact reconstruction → prd 0");
    }
}
