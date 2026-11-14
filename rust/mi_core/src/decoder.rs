//! The streaming decoder (steps 1-6) and the batch helper built on it.

use crate::csv::Row;
use crate::error::{CoreError, Result};
use crate::features::{log_variance, score};
use crate::filter::SosFilter;
use crate::model::{Model, N_CH};
use crate::ring::Ring;
use crate::stim::{validate_code, MarkerTracker, LEFT, RIGHT};
use std::collections::VecDeque;

/// Label of a refused decision: a used channel sat at or beyond `rail_abs` for the whole window.
pub const REFUSED: u32 = 0;

#[derive(Debug, Clone, PartialEq)]
pub struct Decision {
    /// Sample index d at which the decision is made.
    pub sample: usize,
    pub trial: u32,
    /// The marker's movement (1 = left, 2 = right).
    pub true_label: u32,
    pub features: Vec<f64>,
    pub score: f64,
    /// REFUSED (0) if any channel is pinned, else RIGHT (2) if score > 0, else LEFT (1).
    pub label: u32,
    /// Used channels at or beyond `rail_abs` for the whole window, ascending (empty without `rail_abs`).
    pub pinned: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct Decoder {
    model: Model,
    filters: Vec<SosFilter>,
    rings: Vec<Ring>,
    x0: Option<[f64; N_CH]>,
    next: usize,
    markers: MarkerTracker,
    /// Scheduled decisions (d, trial, movement); d only grows, so a queue suffices.
    pending: VecDeque<(usize, u32, u32)>,
    start: usize,
    len: usize,
    /// Per channel: consecutive samples at or beyond `rail_abs`, ending at the latest sample,
    /// saturating at the window length L.
    run: [usize; N_CH],
}

impl Decoder {
    pub fn new(model: Model) -> Self {
        let len = model.window_len();
        Decoder {
            filters: model
                .bands()
                .iter()
                .map(|b| SosFilter::new(&b.sos))
                .collect(),
            rings: model.bands().iter().map(|_| Ring::new(len)).collect(),
            x0: None,
            next: 0,
            markers: MarkerTracker::default(),
            pending: VecDeque::new(),
            start: model.window_start(),
            len,
            run: [0; N_CH],
            model,
        }
    }

    /// Number of samples accepted so far.
    pub fn samples_seen(&self) -> usize {
        self.next
    }

    /// Feed one sample. Invalid input returns an error before any state changes.
    pub fn push(&mut self, sample: &[f64; N_CH], stim: f64) -> Result<Option<Decision>> {
        let i = self.next;
        let code = check_row(i, sample, stim)?;

        self.next += 1;
        if let Some(rail) = self.model.rail_abs() {
            // raw values, before the offset is removed
            for (r, v) in self.run.iter_mut().zip(sample) {
                // saturates at L: pinned only needs run >= L (L <= 2^20, so no overflow)
                *r = if v.abs() >= rail {
                    (*r + 1).min(self.len)
                } else {
                    0
                };
            }
        }
        let x0 = *self.x0.get_or_insert(*sample);
        let mut x = [0.0; N_CH];
        for c in 0..N_CH {
            x[c] = sample[c] - x0[c];
        }
        for (f, r) in self.filters.iter_mut().zip(self.rings.iter_mut()) {
            r.put(i, f.process(&x));
        }
        if let Some(m) = self.markers.observe(code) {
            if m.is_imagery() {
                // start and len are each at most 2^20, so this only overflows after
                // ~2^64 samples on 64-bit targets
                self.pending
                    .push_back((i + self.start + self.len - 1, m.trial, m.movement));
            }
        }
        debug_assert!(
            self.pending.front().map_or(true, |p| p.0 >= i),
            "missed a decision"
        );
        match self.pending.front() {
            Some(&(d, trial, movement)) if d == i => {
                self.pending.pop_front();
                Ok(Some(self.decide(i, trial, movement)))
            }
            _ => Ok(None),
        }
    }

    fn decide(&self, i: usize, trial: u32, movement: u32) -> Decision {
        let mut features = Vec::with_capacity(self.model.lda_w().len());
        for (band, ring) in self.model.bands().iter().zip(&self.rings) {
            log_variance(&band.csp, ring, i, &mut features);
        }
        let s = score(self.model.lda_w(), self.model.lda_b(), &features);
        let pinned: Vec<usize> = if self.model.rail_abs().is_some() {
            let used = self.model.used_channels().iter().copied();
            used.filter(|&c| self.run[c] >= self.len).collect()
        } else {
            Vec::new()
        };
        Decision {
            sample: i,
            trial,
            true_label: movement,
            features,
            score: s,
            label: if !pinned.is_empty() {
                REFUSED
            } else if s > 0.0 {
                RIGHT
            } else {
                LEFT
            },
            pinned,
        }
    }
}

/// The contract's input checks for one row, EEG first, then stim. Returns the stim code.
/// The single source of these rules for both streaming and batch decoding.
fn check_row(index: usize, sample: &[f64; N_CH], stim: f64) -> Result<u32> {
    if sample.iter().any(|v| !v.is_finite()) {
        return Err(CoreError::Sample {
            index,
            reason: "NaN or infinite value".into(),
        });
    }
    validate_code(stim).map_err(|reason| CoreError::Stim { index, reason })
}

/// Batch decoding: validates every row first (rejecting the whole stream), then streams.
pub fn decode(model: &Model, rows: &[Row]) -> Result<Vec<Decision>> {
    for (i, row) in rows.iter().enumerate() {
        check_row(i, &row.sample, row.stim)?;
    }
    let mut dec = Decoder::new(model.clone());
    let mut out = Vec::new();
    for row in rows {
        if let Some(d) = dec.push(&row.sample, row.stim)? {
            out.push(d);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rail_model() -> Model {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../golden/model.json");
        let mut v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        v["rail_abs"] = serde_json::json!(1e6);
        Model::from_json(&v.to_string()).unwrap()
    }

    #[test]
    fn pinned_run_counters_saturate_at_the_window_length() {
        let mut dec = Decoder::new(rail_model());
        let len = dec.len;
        let mut s = [5e5; N_CH];
        s[2] = 1e6;
        for _ in 0..2 * len + 7 {
            dec.push(&s, 0.0).unwrap();
        }
        assert_eq!(dec.run, [0, 0, len, 0, 0, 0, 0, 0]);
        dec.push(&[5e5; N_CH], 0.0).unwrap();
        assert_eq!(dec.run, [0; N_CH]);
    }
}
