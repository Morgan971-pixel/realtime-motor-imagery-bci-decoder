//! Real-time motor-imagery decoder core: inference only, following `docs/model-format.md`.
//!
//! Training stays in Python (`mi_ref`); this crate loads `model.json`, runs the stream
//! sample by sample and must match the golden files within the contract tolerances.

pub mod csv;
pub mod decoder;
pub mod error;
pub mod features;
pub mod filter;
pub mod model;
pub mod replay;
pub mod ring;
pub mod stim;
pub mod timing;

#[cfg(feature = "python")]
mod python;

pub use decoder::{decode, Decision, Decoder, REFUSED};
pub use error::{CoreError, Result};
pub use model::Model;
