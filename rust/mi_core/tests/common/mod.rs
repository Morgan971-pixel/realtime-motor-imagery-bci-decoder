//! Shared helpers: paths to the committed golden files and the contract tolerances.

#![allow(dead_code)]

use std::path::PathBuf;

pub const FILTER_ABS: f64 = 1e-9;
pub const FEATURE_REL: f64 = 1e-9;
pub const SCORE_ABS: f64 = 1e-7;

pub fn golden(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../golden")
        .join(name)
}

/// Relative check for values well away from 0 (features). Not for scores, which can be ~0.
pub fn assert_rel(got: f64, want: f64, rel: f64, what: &str) {
    let tol = rel * want.abs().max(f64::MIN_POSITIVE);
    assert!((got - want).abs() <= tol, "{what}: got {got}, want {want}");
}
