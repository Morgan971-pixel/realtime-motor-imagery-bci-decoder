//! Strict `model.json` loader: every load error in the contract is an `Err`, never a fallback.
//!
//! serde does most of the type checking: `deny_unknown_fields` rejects unknown keys and the
//! derive rejects repeated keys; numbers written as strings or booleans fail to deserialize;
//! `require_objects` rejects arrays standing in for objects (serde would accept them);
//! fixed-size arrays reject rows of the wrong width; JSON has no NaN, and serde_json refuses
//! numbers that overflow f64 (such as 1e400). `validate` checks the value rules. The one
//! optional key, `rail_abs`, may be left out but never written as `null`.

use crate::error::{CoreError, Result};
use crate::timing::to_samples;
use serde::Deserialize;

pub const N_CH: usize = 8;
pub const CHANNELS: [&str; N_CH] = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"];
/// Window start and length may each span at most 2^20 samples (about 70 min at 250 Hz).
pub const MAX_WINDOW_SAMPLES: f64 = 1_048_576.0;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawModel {
    schema_version: u32,
    fs: f64,
    channels: Vec<String>,
    offset: String,
    window: RawWindow,
    bands: Vec<RawBand>,
    lda: RawLda,
    labels: RawLabels,
    #[serde(default, deserialize_with = "present_number")]
    rail_abs: Option<f64>,
}

/// A key that may be left out (None) but, when present, must be a JSON number. serde's own
/// Option would read `null` as None, which the contract refuses.
fn present_number<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> std::result::Result<Option<f64>, D::Error> {
    f64::deserialize(d).map(Some)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawWindow {
    start_s: f64,
    len_s: f64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawBand {
    low_hz: f64,
    high_hz: f64,
    sos: Vec<[f64; 6]>,
    csp: Vec<[f64; N_CH]>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawLda {
    w: Vec<f64>,
    b: f64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawLabels {
    score_le_0: u32,
    score_gt_0: u32,
}

#[derive(Debug, Clone)]
pub struct Band {
    /// Second-order sections `[b0, b1, b2, a0, a1, a2]`, applied in order.
    pub sos: Vec<[f64; 6]>,
    /// CSP rows, each a weight per channel.
    pub csp: Vec<[f64; N_CH]>,
}

/// A validated model. Fields are private: `Model::from_json` / `from_path` are the only constructors.
#[derive(Debug, Clone)]
pub struct Model {
    fs: f64,
    window_start_s: f64,
    window_len_s: f64,
    bands: Vec<Band>,
    lda_w: Vec<f64>,
    lda_b: f64,
    rail_abs: Option<f64>,
    used_channels: Vec<usize>,
}

fn invalid(msg: impl Into<String>) -> CoreError {
    CoreError::Model(msg.into())
}

impl Model {
    pub fn from_json(text: &str) -> Result<Model> {
        let raw: RawModel = serde_json::from_str(text).map_err(|e| invalid(e.to_string()))?;
        require_objects(text)?;
        validate(raw)
    }

    pub fn from_path(path: impl AsRef<std::path::Path>) -> Result<Model> {
        Model::from_json(&std::fs::read_to_string(path)?)
    }

    /// Sample rate in Hz.
    pub fn fs(&self) -> f64 {
        self.fs
    }

    /// Bands in order: each band's filter sections and CSP rows.
    pub fn bands(&self) -> &[Band] {
        &self.bands
    }

    /// LDA weights, band-major (band 0's CSP rows first).
    pub fn lda_w(&self) -> &[f64] {
        &self.lda_w
    }

    /// LDA bias.
    pub fn lda_b(&self) -> f64 {
        self.lda_b
    }

    /// Raw value at or beyond which a channel counts as pinned; None turns the check off.
    pub fn rail_abs(&self) -> Option<f64> {
        self.rail_abs
    }

    /// Channels with a nonzero CSP weight in some band, ascending.
    pub fn used_channels(&self) -> &[usize] {
        &self.used_channels
    }

    /// Window length L in samples.
    pub fn window_len(&self) -> usize {
        to_samples(self.window_len_s, self.fs)
    }

    /// Window start offset after the imagery onset, in samples.
    pub fn window_start(&self) -> usize {
        to_samples(self.window_start_s, self.fs)
    }
}

/// serde fills a struct from a JSON array too (fields in order); the contract requires objects.
/// The typed parse above runs first because it is the one that rejects repeated keys.
fn require_objects(text: &str) -> Result<()> {
    let v: serde_json::Value = serde_json::from_str(text).map_err(|e| invalid(e.to_string()))?;
    let object = |value: &serde_json::Value, at: &str| {
        if value.is_object() {
            Ok(())
        } else {
            Err(invalid(format!("{at} must be a JSON object")))
        }
    };
    object(&v, "model")?;
    for key in ["window", "lda", "labels"] {
        object(&v[key], key)?;
    }
    for (i, band) in v["bands"].as_array().into_iter().flatten().enumerate() {
        object(band, &format!("bands[{i}]"))?;
    }
    Ok(())
}

fn validate(r: RawModel) -> Result<Model> {
    if r.schema_version != 1 {
        return Err(invalid(format!(
            "unsupported schema_version {}",
            r.schema_version
        )));
    }
    if r.offset != "first_sample" {
        return Err(invalid(format!("unsupported offset rule {:?}", r.offset)));
    }
    if (r.labels.score_le_0, r.labels.score_gt_0) != (1, 2) {
        return Err(invalid(
            "labels must be {\"score_le_0\": 1, \"score_gt_0\": 2}",
        ));
    }
    if !r.channels.iter().map(String::as_str).eq(CHANNELS) {
        return Err(invalid(format!(
            "channels must be {CHANNELS:?}, got {:?}",
            r.channels
        )));
    }
    if !(r.fs.is_finite() && r.fs > 0.0) {
        return Err(invalid(format!("fs must be finite and > 0, got {}", r.fs)));
    }
    let (start, len) = (r.window.start_s, r.window.len_s);
    if !(start.is_finite() && len.is_finite() && start >= 0.0 && len > 0.0) {
        return Err(invalid(format!(
            "window needs start_s >= 0 and len_s > 0, both finite; got start_s {start}, len_s {len}"
        )));
    }
    for (name, seconds) in [("start_s", start), ("len_s", len)] {
        // checked before any cast to usize; the product overflows to inf for huge fs
        let span = seconds * r.fs;
        if !span.is_finite() || span > MAX_WINDOW_SAMPLES {
            return Err(invalid(format!(
                "window {name} spans more than {MAX_WINDOW_SAMPLES} samples"
            )));
        }
    }
    if to_samples(len, r.fs) < 2 {
        return Err(invalid("window must span at least 2 samples"));
    }
    let first = r
        .bands
        .first()
        .ok_or_else(|| invalid("need at least one band"))?;
    let rows = first.csp.len();
    if rows == 0 || rows % 2 != 0 {
        return Err(invalid(format!(
            "csp needs an even, nonzero row count, got {rows}"
        )));
    }
    let mut bands = Vec::with_capacity(r.bands.len());
    for (i, b) in r.bands.into_iter().enumerate() {
        if !(b.low_hz.is_finite()
            && b.high_hz.is_finite()
            && 0.0 < b.low_hz
            && b.low_hz < b.high_hz
            && b.high_hz < r.fs / 2.0)
        {
            return Err(invalid(format!(
                "band {i} must satisfy 0 < low < high < fs/2; got low {}, high {}, fs {}",
                b.low_hz, b.high_hz, r.fs
            )));
        }
        if b.sos.is_empty() || b.sos.iter().flatten().any(|v| !v.is_finite()) {
            return Err(invalid(format!(
                "band {i}: sos must be non-empty and finite"
            )));
        }
        if let Some(k) = b.sos.iter().position(|s| s[3] != 1.0) {
            return Err(invalid(format!("band {i}, section {k}: a0 must equal 1")));
        }
        if b.csp.len() != rows || b.csp.iter().flatten().any(|v| !v.is_finite()) {
            return Err(invalid(format!(
                "band {i}: csp must have {rows} finite rows"
            )));
        }
        bands.push(Band {
            sos: b.sos,
            csp: b.csp,
        });
    }
    if r.lda.w.len() != bands.len() * rows
        || r.lda.w.iter().any(|v| !v.is_finite())
        || !r.lda.b.is_finite()
    {
        return Err(invalid(format!(
            "lda.w must be {} finite numbers and lda.b finite",
            bands.len() * rows
        )));
    }
    if let Some(rail) = r.rail_abs {
        if !(rail.is_finite() && rail > 0.0) {
            return Err(invalid(format!(
                "rail_abs must be finite and > 0, got {rail}"
            )));
        }
    }
    let used_channels = (0..N_CH)
        .filter(|&c| bands.iter().any(|b| b.csp.iter().any(|row| row[c] != 0.0)))
        .collect();
    Ok(Model {
        fs: r.fs,
        window_start_s: start,
        window_len_s: len,
        bands,
        lda_w: r.lda.w,
        lda_b: r.lda.b,
        rail_abs: r.rail_abs,
        used_channels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn golden_model_loads_with_the_expected_shape() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../golden/model.json");
        let m = Model::from_path(path).unwrap();
        assert_eq!(m.fs(), 250.0);
        assert_eq!((m.window_start(), m.window_len()), (125, 500));
        assert_eq!(m.bands().len(), 4);
        assert!(m
            .bands()
            .iter()
            .all(|b| b.csp.len() == 4 && b.sos.len() == 4));
        assert_eq!(m.lda_w().len(), 16);
    }

    /// The integer spelling -0 is negative zero, the correctly rounded double of its text (C++ and
    /// Python once read it as +0; found by the Hypothesis property on re-spelled numbers).
    #[test]
    fn negative_zero_written_as_an_integer_keeps_its_sign() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../golden/model.json");
        let text = std::fs::read_to_string(path).unwrap();
        let text = text.replace("\"b\": -19.391831086424165", "\"b\": -0");
        assert!(text.contains("\"b\": -0\n"));
        let m = Model::from_json(&text).unwrap();
        assert_eq!(m.lda_b(), 0.0);
        assert!(m.lda_b().is_sign_negative());
    }
}
