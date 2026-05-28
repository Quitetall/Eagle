//! Reference codec adapters.
//!
//! An adapter wraps a codec behind the uniform [`Codec`] byte interface
//! so the harness can grade any codec the same way. Three reference
//! adapters ship so the suite has something to run out of the box:
//!
//! - [`Store`] — identity/raw passthrough. The CR≈1.0 lossless baseline.
//! - [`Gzip`]  — pure-Rust miniz_oxide via `flate2` (no system dep), the
//!   always-available compressed lossless reference.
//! - [`Zstd`]  — optional, behind `#[cfg(feature = "zstd")]`, so the
//!   default CI build carries no system dependency.
//!
//! ## Codec interface
//!
//! A [`Codec`] takes a per-channel signal (`&[Vec<i64>]`, one `Vec<i64>`
//! of samples per channel) plus the sample rate, and produces an opaque
//! `Vec<u8>` blob; [`Codec::decode`] inverts that. The reference adapters
//! are all *declared lossless* and round-trip the channel layout
//! bit-exactly.
//!
//! ## Reference serialization
//!
//! The reference adapters share one tiny, explicit, byte-exact container
//! (see [`serialize`] / [`deserialize`]): a fixed header followed by the
//! samples. Layout (all integers little-endian):
//!
//! ```text
//! magic   : 4 bytes  = b"LQS0"
//! n_chan  : u32       = number of channels
//! per chan: u32 len, then `len` × i64 samples
//! ```
//!
//! The sample rate `fs` is *not* part of the byte stream: it is metadata
//! the harness already knows, and the reference codecs reconstruct the
//! integer samples losslessly without it. A real codec is free to embed
//! `fs` in its own blob; the trait simply makes it available.

use std::io::Write;

use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;

/// Magic prefix for the reference serialization container.
const MAGIC: &[u8; 4] = b"LQS0";

/// The codec-agnostic interface every benchmarked codec implements.
///
/// `signal` is one `Vec<i64>` of samples per channel; `fs` is the sample
/// rate in Hz (metadata the codec may use or ignore). [`encode`] returns
/// an opaque blob and [`decode`] reconstructs the per-channel samples.
///
/// A codec that advertises [`declared_lossless`] `== true` is *claiming*
/// bit-exact reconstruction; the harness verifies that claim against the
/// L-tier gate. The claim is what is graded — it is not assumed true.
///
/// [`encode`]: Codec::encode
/// [`decode`]: Codec::decode
/// [`declared_lossless`]: Codec::declared_lossless
pub trait Codec {
    /// Short, stable identifier used in reports (e.g. `"store"`).
    fn name(&self) -> &str;

    /// Whether the codec *claims* bit-exact reconstruction.
    fn declared_lossless(&self) -> bool;

    /// Compress a per-channel integer signal into an opaque blob.
    fn encode(&self, signal: &[Vec<i64>], fs: f64) -> Vec<u8>;

    /// Reconstruct the per-channel integer signal from a blob.
    fn decode(&self, blob: &[u8]) -> Vec<Vec<i64>>;
}

/// Serialize a per-channel integer signal to the reference container.
///
/// Deterministic and byte-exact: the same input always produces the same
/// bytes, so two backends can be compared for byte-equality. Integers are
/// little-endian. See the module docs for the layout.
pub fn serialize(signal: &[Vec<i64>]) -> Vec<u8> {
    // Pre-size: magic + n_chan, then per channel a u32 length + the samples.
    let total_samples: usize = signal.iter().map(|c| c.len()).sum();
    let mut out = Vec::with_capacity(4 + 4 + signal.len() * 4 + total_samples * 8);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(signal.len() as u32).to_le_bytes());
    for chan in signal {
        out.extend_from_slice(&(chan.len() as u32).to_le_bytes());
        for &s in chan {
            out.extend_from_slice(&s.to_le_bytes());
        }
    }
    out
}

/// Inverse of [`serialize`].
///
/// Returns an empty `Vec` if the buffer is malformed (bad magic, or a
/// truncated header / sample run). The reference adapters only ever feed
/// this their own [`serialize`] output, so a malformed buffer means a
/// corrupt blob rather than a recoverable state; the harness's L-tier
/// gate then sees a length/value mismatch and fails the lossless claim,
/// which is the correct outcome.
pub fn deserialize(buf: &[u8]) -> Vec<Vec<i64>> {
    // Header: 4-byte magic + 4-byte channel count.
    if buf.len() < 8 || &buf[..4] != MAGIC {
        return Vec::new();
    }
    let n_chan = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]) as usize;
    let mut pos = 8usize;
    let mut out = Vec::with_capacity(n_chan);
    for _ in 0..n_chan {
        // Per-channel sample count.
        if pos + 4 > buf.len() {
            return Vec::new();
        }
        let len = u32::from_le_bytes([buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3]]) as usize;
        pos += 4;
        // `len` × i64 samples must fit in the remaining buffer.
        let need = len.checked_mul(8);
        match need {
            Some(bytes) if pos + bytes <= buf.len() => {}
            _ => return Vec::new(),
        }
        let mut chan = Vec::with_capacity(len);
        for _ in 0..len {
            let mut b = [0u8; 8];
            b.copy_from_slice(&buf[pos..pos + 8]);
            chan.push(i64::from_le_bytes(b));
            pos += 8;
        }
        out.push(chan);
    }
    out
}

/// Gzip-compress a byte buffer with the pure-Rust backend.
///
/// This is the [`Gzip`] adapter's compression primitive. It never fails
/// for in-memory buffers (miniz_oxide writes to a `Vec`); the `expect`
/// guards a genuinely unreachable I/O error on an in-memory writer.
pub fn gzip_compress(data: &[u8]) -> Vec<u8> {
    let mut enc = GzEncoder::new(Vec::new(), Compression::default());
    enc.write_all(data).expect("gzip write to Vec is infallible");
    enc.finish().expect("gzip finish on Vec is infallible")
}

/// Gunzip a byte buffer produced by [`gzip_compress`].
///
/// Returns an empty `Vec` if the input is not valid gzip; as with
/// [`deserialize`] this surfaces a corrupt blob to the L-tier gate as a
/// failed lossless claim rather than a panic.
pub fn gzip_decompress(data: &[u8]) -> Vec<u8> {
    use std::io::Read;
    let mut dec = GzDecoder::new(data);
    let mut out = Vec::new();
    match dec.read_to_end(&mut out) {
        Ok(_) => out,
        Err(_) => Vec::new(),
    }
}

#[cfg(feature = "zstd")]
/// Zstd-compress a byte buffer (level 19). Only present under the
/// `zstd` feature so the default CI build carries no system dependency.
pub fn zstd_compress(data: &[u8]) -> Vec<u8> {
    zstd::stream::encode_all(data, 19).expect("zstd encode of in-memory buffer")
}

#[cfg(feature = "zstd")]
/// Zstd-decompress a buffer produced by [`zstd_compress`]. Returns an
/// empty `Vec` on malformed input (see [`gzip_decompress`]).
pub fn zstd_decompress(data: &[u8]) -> Vec<u8> {
    zstd::stream::decode_all(data).unwrap_or_default()
}

/// Identity/raw passthrough adapter — the CR≈1.0 lossless baseline.
///
/// `encode` is just [`serialize`]; `decode` is just [`deserialize`]. No
/// compression happens, so this is the reference point against which
/// every other codec's compression ratio is measured.
#[derive(Clone, Copy, Debug, Default)]
pub struct Store;

impl Codec for Store {
    fn name(&self) -> &str {
        "store"
    }

    fn declared_lossless(&self) -> bool {
        true
    }

    fn encode(&self, signal: &[Vec<i64>], _fs: f64) -> Vec<u8> {
        serialize(signal)
    }

    fn decode(&self, blob: &[u8]) -> Vec<Vec<i64>> {
        deserialize(blob)
    }
}

/// Gzip adapter — pure-Rust miniz_oxide via `flate2`, always available.
///
/// `encode` = [`serialize`] then [`gzip_compress`]; `decode` =
/// [`gzip_decompress`] then [`deserialize`]. Bit-exact lossless.
#[derive(Clone, Copy, Debug, Default)]
pub struct Gzip;

impl Codec for Gzip {
    fn name(&self) -> &str {
        "gzip"
    }

    fn declared_lossless(&self) -> bool {
        true
    }

    fn encode(&self, signal: &[Vec<i64>], _fs: f64) -> Vec<u8> {
        gzip_compress(&serialize(signal))
    }

    fn decode(&self, blob: &[u8]) -> Vec<Vec<i64>> {
        deserialize(&gzip_decompress(blob))
    }
}

/// Zstd adapter — optional, compiled in only under the `zstd` feature.
///
/// `encode` = [`serialize`] then [`zstd_compress`]; `decode` =
/// [`zstd_decompress`] then [`deserialize`]. Bit-exact lossless.
#[cfg(feature = "zstd")]
#[derive(Clone, Copy, Debug, Default)]
pub struct Zstd;

#[cfg(feature = "zstd")]
impl Codec for Zstd {
    fn name(&self) -> &str {
        "zstd"
    }

    fn declared_lossless(&self) -> bool {
        true
    }

    fn encode(&self, signal: &[Vec<i64>], _fs: f64) -> Vec<u8> {
        zstd_compress(&serialize(signal))
    }

    fn decode(&self, blob: &[u8]) -> Vec<Vec<i64>> {
        deserialize(&zstd_decompress(blob))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A known multi-channel i64 signal that exercises sign, zero, the
    /// i64 extremes, a ragged channel length, and an empty channel.
    fn fixture() -> Vec<Vec<i64>> {
        vec![
            vec![0, 1, -1, 1000, -1000, i64::MAX, i64::MIN],
            vec![-42, 42, 0, 7],
            vec![], // empty channel must survive the round trip
            vec![123_456_789_012, -987_654_321_098],
        ]
    }

    /// Round-trip helper: assert the codec reconstructs the fixture
    /// exactly and (for the declared-lossless reference adapters) keeps
    /// the lossless claim honest.
    fn assert_roundtrip<C: Codec>(codec: C) {
        let signal = fixture();
        let blob = codec.encode(&signal, 256.0);
        let back = codec.decode(&blob);
        assert_eq!(
            back,
            signal,
            "{} failed bit-exact round trip",
            codec.name()
        );
        assert!(
            codec.declared_lossless(),
            "{} reference adapter should declare lossless",
            codec.name()
        );
    }

    #[test]
    fn serialize_deserialize_is_exact() {
        let signal = fixture();
        let bytes = serialize(&signal);
        assert_eq!(deserialize(&bytes), signal);
    }

    #[test]
    fn serialize_is_deterministic() {
        // Byte-for-byte stable across calls — the property the
        // cross-backend byte-equality gate relies on.
        let signal = fixture();
        assert_eq!(serialize(&signal), serialize(&signal));
    }

    #[test]
    fn deserialize_rejects_bad_magic() {
        assert!(deserialize(b"XXXX\x00\x00\x00\x00").is_empty());
        assert!(deserialize(&[]).is_empty());
        // Truncated header (claims 1 channel, no length follows).
        let mut buf = MAGIC.to_vec();
        buf.extend_from_slice(&1u32.to_le_bytes());
        assert!(deserialize(&buf).is_empty());
    }

    #[test]
    fn store_roundtrips() {
        assert_roundtrip(Store);
        // Store does not compress: blob is exactly the serialization.
        let signal = fixture();
        assert_eq!(Store.encode(&signal, 256.0), serialize(&signal));
    }

    #[test]
    fn gzip_roundtrips() {
        assert_roundtrip(Gzip);
    }

    #[test]
    fn gzip_decompress_rejects_garbage() {
        // Not valid gzip => empty, then empty deserialize => empty signal.
        assert!(gzip_decompress(b"not gzip data").is_empty());
        assert!(Gzip.decode(b"not gzip data").is_empty());
    }

    #[cfg(feature = "zstd")]
    #[test]
    fn zstd_roundtrips() {
        assert_roundtrip(Zstd);
    }

    #[test]
    fn empty_signal_roundtrips() {
        // Zero channels is a valid (degenerate) signal.
        let empty: Vec<Vec<i64>> = Vec::new();
        assert_eq!(Store.decode(&Store.encode(&empty, 256.0)), empty);
        assert_eq!(Gzip.decode(&Gzip.encode(&empty, 256.0)), empty);
    }
}
