//! Property tests against independent oracles: the marker rule and the features are
//! recomputed here from scratch, without the crate's MarkerTracker, Ring or log_variance.

mod common;

use common::{golden, FEATURE_REL, SCORE_ABS};
use mi_core::csv::Row;
use mi_core::filter::SosFilter;
use mi_core::{decode, Decision, Decoder, Model};
use proptest::prelude::*;
use std::sync::OnceLock;

// 130 and 133 are phase 3 with a movement that is neither left nor right: never a decision
const CODES: [u32; 10] = [131, 132, 121, 151, 111, 141, 232, 331, 130, 133];

fn model() -> &'static Model {
    static MODEL: OnceLock<Model> = OnceLock::new();
    MODEL.get_or_init(|| Model::from_path(golden("model.json")).unwrap())
}

/// Rows with a large per-channel offset (to exercise offset removal) and codes drawn per sample.
fn rows_with(code: BoxedStrategy<u32>) -> impl Strategy<Value = Vec<Row>> {
    (
        prop::array::uniform8(-1e4f64..1e4),
        prop::collection::vec((prop::array::uniform8(-500.0f64..500.0), code), 600..1600),
    )
        .prop_map(|(offset, samples)| {
            samples
                .into_iter()
                .map(|(noise, code)| {
                    let mut sample = noise;
                    for (s, o) in sample.iter_mut().zip(offset) {
                        *s += o;
                    }
                    Row {
                        sample,
                        stim: code as f64,
                    }
                })
                .collect()
        })
}

/// Each stream is either dense (a code on most samples: runs, A,B,A, overlapping windows)
/// or sparse (about one code in 61 samples: long silent gaps, often nothing pending).
fn stream() -> impl Strategy<Value = Vec<Row>> {
    let dense = prop_oneof![1 => Just(0u32), 8 => prop::sample::select(&CODES[..])].boxed();
    let sparse = prop_oneof![60 => Just(0u32), 1 => prop::sample::select(&CODES[..])].boxed();
    prop_oneof![rows_with(dense), rows_with(sparse)]
}

/// The contract's marker rule, written out independently: (decision sample, trial, movement).
fn expected_decisions(rows: &[Row], offset: usize) -> Vec<(usize, u32, u32)> {
    let mut prev = 0u32;
    let mut out = Vec::new();
    for (i, r) in rows.iter().enumerate() {
        let c = r.stim as u32;
        let (phase, movement) = ((c / 10) % 10, c % 10);
        if c != 0
            && c != prev
            && phase == 3
            && (movement == 1 || movement == 2)
            && i + offset < rows.len()
        {
            out.push((i + offset, c / 100, movement));
        }
        prev = c;
    }
    out
}

/// Features recomputed from scratch: filter the whole stream, slice the window, plain loops.
fn oracle_features(model: &Model, rows: &[Row], d: usize) -> Vec<f64> {
    let len = model.window_len();
    let x0 = rows[0].sample;
    let mut features = Vec::new();
    for band in model.bands() {
        let mut f = SosFilter::new(&band.sos);
        let filtered: Vec<[f64; 8]> = rows[..=d]
            .iter()
            .map(|r| {
                let mut x = r.sample;
                for c in 0..8 {
                    x[c] -= x0[c];
                }
                f.process(&x)
            })
            .collect();
        let window = &filtered[d + 1 - len..=d];
        for w in &band.csp {
            let z: Vec<f64> = window
                .iter()
                .map(|x| (0..8).map(|c| w[c] * x[c]).sum())
                .collect();
            let mean = z.iter().sum::<f64>() / len as f64;
            let var = z.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / len as f64;
            features.push(var.max(1e-12).ln());
        }
    }
    features
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    #[test]
    fn decisions_match_independent_oracles(rows in stream()) {
        let model = model();
        let offset = model.window_start() + model.window_len() - 1;
        let got: Vec<Decision> = decode(model, &rows).unwrap();

        let placed: Vec<_> = got.iter().map(|d| (d.sample, d.trial, d.true_label)).collect();
        prop_assert_eq!(placed, expected_decisions(&rows, offset));

        for d in &got {
            let want = oracle_features(model, &rows, d.sample);
            prop_assert_eq!(d.features.len(), want.len());
            for (g, w) in d.features.iter().zip(&want) {
                prop_assert!((g - w).abs() <= (FEATURE_REL * w.abs()).max(1e-12), "feature {} vs {}", g, w);
            }
            let score: f64 = model.lda_w().iter().zip(&want).map(|(a, b)| a * b).sum::<f64>() + model.lda_b();
            prop_assert!((d.score - score).abs() <= SCORE_ABS);
            prop_assert_eq!(d.label, if d.score > 0.0 { 2 } else { 1 });
        }
    }

    #[test]
    fn streaming_equals_batch(rows in stream()) {
        let model = model();
        let mut dec = Decoder::new(model.clone());
        let streamed: Vec<_> = rows.iter().filter_map(|r| dec.push(&r.sample, r.stim).unwrap()).collect();
        prop_assert_eq!(streamed, decode(model, &rows).unwrap());
    }
}
