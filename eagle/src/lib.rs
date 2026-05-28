//! Eagle — LamQuant's benchmarking tool.
//!
//! Eagle is a *consumer* of the vendor-neutral [`lqs`] standard. LQS
//! defines the agnostic grade (PRD/PRDN/R/SNR/CR + per-band, tiers
//! L/C/M/A) and ships only neutral reference adapters. Eagle adds the
//! LamQuant-specific pieces:
//!
//!   - [`adapters_lamquant`] — a [`lqs::adapter::Codec`] adapter that
//!     shells to the real `lml` lossless codec binary.
//!   - (future) corpus runners over real EEG datasets, LamQuant report
//!     rollups, and the `-m internal` neural-introspection suite.
//!
//! The split is deliberate: anyone can `cargo add lqs` to grade their
//! own codec; Eagle is how LamQuant grades *its* codecs with LQS.

pub mod adapters_lamquant;

pub use adapters_lamquant::LamQuantLossless;
