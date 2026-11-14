//! The committed golden files are the contract: the Rust core must reproduce them.

mod common;

use common::*;
use mi_core::csv::read_csv;
use mi_core::filter::SosFilter;
use mi_core::{decode, Decoder, Model};
use serde::Deserialize;

#[derive(Deserialize)]
struct Expected {
    stream: Case,
    edge_stream: Case,
    rail_stream: Case,
}

#[derive(Deserialize)]
struct Case {
    band0_filtered_head: Vec<[f64; 8]>,
    filtered_tail_per_band: Vec<Vec<[f64; 8]>>,
    decisions: Vec<Dec>,
}

#[derive(Deserialize)]
struct Dec {
    sample: usize,
    trial: u32,
    true_label: u32,
    features: Vec<f64>,
    score: f64,
    label: u32,
    pinned: Vec<usize>,
}

fn expected() -> Expected {
    serde_json::from_str(&std::fs::read_to_string(golden("expected.json")).unwrap()).unwrap()
}

/// The model each golden stream is decoded with.
fn model_for(name: &str) -> Model {
    let file = if name == "rail_stream" {
        "rail_model.json"
    } else {
        "model.json"
    };
    Model::from_path(golden(file)).unwrap()
}

fn check_stream(name: &str, case: &Case) {
    let model = model_for(name);
    let rows = read_csv(golden(&format!("{name}.csv"))).unwrap();

    // Filtered samples: offset removal (step 1) then each band's filter (step 2).
    let x0 = rows[0].sample;
    for (b, band) in model.bands().iter().enumerate() {
        let mut f = SosFilter::new(&band.sos);
        let out: Vec<[f64; 8]> = rows
            .iter()
            .map(|r| {
                let mut x = r.sample;
                for c in 0..8 {
                    x[c] -= x0[c];
                }
                f.process(&x)
            })
            .collect();
        let tail = &case.filtered_tail_per_band[b];
        for (got, want) in out[out.len() - tail.len()..].iter().zip(tail) {
            for c in 0..8 {
                assert!(
                    (got[c] - want[c]).abs() <= FILTER_ABS,
                    "{name} band {b} tail"
                );
            }
        }
        if b == 0 {
            for (got, want) in out.iter().zip(&case.band0_filtered_head) {
                for c in 0..8 {
                    assert!((got[c] - want[c]).abs() <= FILTER_ABS, "{name} band 0 head");
                }
            }
        }
    }

    // Decisions: exact sample, trial, labels and count; features rel 1e-9; score abs 1e-7.
    let got = decode(&model, &rows).unwrap();
    assert_eq!(got.len(), case.decisions.len(), "{name}: decision count");
    for (g, e) in got.iter().zip(&case.decisions) {
        assert_eq!(
            (g.sample, g.trial, g.true_label, g.label, &g.pinned),
            (e.sample, e.trial, e.true_label, e.label, &e.pinned),
            "{name}"
        );
        assert_eq!(g.features.len(), e.features.len());
        for (gf, ef) in g.features.iter().zip(&e.features) {
            assert_rel(*gf, *ef, FEATURE_REL, name);
        }
        assert!(
            (g.score - e.score).abs() <= SCORE_ABS,
            "{name}: score {} vs {}",
            g.score,
            e.score
        );
    }
}

#[test]
fn reproduces_the_golden_stream() {
    check_stream("stream", &expected().stream);
}

#[test]
fn reproduces_the_golden_edge_stream() {
    check_stream("edge_stream", &expected().edge_stream);
}

#[test]
fn reproduces_the_golden_rail_stream() {
    check_stream("rail_stream", &expected().rail_stream);
}

#[test]
fn streaming_push_reproduces_batch_on_every_golden_stream() {
    for name in ["stream", "edge_stream", "rail_stream"] {
        let rows = read_csv(golden(&format!("{name}.csv"))).unwrap();
        let mut dec = Decoder::new(model_for(name));
        let streamed: Vec<_> = rows
            .iter()
            .filter_map(|r| dec.push(&r.sample, r.stim).unwrap())
            .collect();
        assert_eq!(streamed, decode(&model_for(name), &rows).unwrap(), "{name}");
    }
}
