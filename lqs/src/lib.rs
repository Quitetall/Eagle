//! # LQS — the vendor-neutral EEG codec benchmark standard
//!
//! LQS (LamQuant Quality Standard, now vendor-neutral) is an open
//! benchmark for evaluating EEG compression quality. Any codec —
//! neural, classical, or hybrid — can declare compliance with an LQS
//! quality tier by passing the standard test suite against a standard
//! holdout corpus. There are no self-reported numbers and no
//! cherry-picked patients: standardized holdout, standardized metrics,
//! deterministic pass/fail.
//!
//! ## Tiers
//!
//! | Tier | Name       | Intent                                        |
//! |------|------------|-----------------------------------------------|
//! | L    | Lossless   | bit-exact reconstruction (PRD == 0 exactly)   |
//! | C    | Clinical   | a neurologist cannot distinguish recon        |
//! | M    | Monitoring | automated analysis preserved                  |
//! | A    | Alerting   | event detection preserved                     |
//!
//! A codec declares e.g. "I am LQS-C compliant at CR=42:1" and the test
//! harness verifies or rejects the claim.
//!
//! ## Why Rust is now canonical
//!
//! The Rust implementation in this crate is the canonical LQS spec: it
//! is faster to run in CI and pre-commit hooks than the legacy Python
//! reference, and it is the version the grading gate enforces. The
//! Python module remains as a readable reference but defers to this
//! crate for the authoritative thresholds and gate logic.
//!
//! ## Modules
//!
//! - [`levels`]  — the spec: tier table + the `grade` gate logic.
//! - [`metrics`] — canonical metric formulas (PRD, Pearson R, SNR, CR…).
//! - [`bands`]   — per-EEG-band fidelity helpers (fill phase).
//! - [`adapter`] — reference codec adapters (gzip, optional zstd).
//! - [`harness`] — the compliance test runner (fill phase).
//! - [`report`]  — JSON / badge reporting (fill phase).

pub mod adapter;
pub mod adapters_lamquant;
pub mod bands;
pub mod edf;
pub mod harness;
pub mod levels;
pub mod metrics;
pub mod report;
pub mod suites;
