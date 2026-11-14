//! One error type for the whole crate. Every rejection in the contract maps to a variant.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum CoreError {
    /// `model.json` breaks a load rule (shape, type, value, unknown or repeated key).
    #[error("invalid model: {0}")]
    Model(String),
    /// A stim value breaks the input contract (step 4).
    #[error("invalid stim at sample {index}: {reason}")]
    Stim { index: usize, reason: String },
    /// An EEG value is NaN or infinite.
    #[error("invalid EEG sample at sample {index}: {reason}")]
    Sample { index: usize, reason: String },
    /// A recording file does not have the club CSV layout.
    #[error("invalid CSV: {0}")]
    Csv(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, CoreError>;
