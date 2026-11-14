//! Every invalid stream in golden/invalid/manifest.json must be rejected at its bad row and
//! leave no trace; every valid boundary stream must be accepted.

mod common;

use common::golden;
use mi_core::csv::{read_csv, Row};
use mi_core::{decode, CoreError, Decision, Decoder, Model};
use serde::Deserialize;
use serde_json::Value;

#[derive(Deserialize)]
struct Manifest {
    streams: Vec<StreamCase>,
    valid_streams: Vec<ValidCase>,
    rail_streams: Vec<RailCase>,
    valid_models: Vec<ValidModel>,
}

/// Rows run-length encoded as [count, row]; `rejected` rows (indices into the expanded rows)
/// must fail with their kind; `decisions` are [sample, label, pinned].
#[derive(Deserialize)]
struct RailCase {
    name: String,
    rows: Vec<(usize, Vec<Value>)>,
    rejected: Vec<Rejected>,
    decisions: Vec<(usize, u32, Vec<usize>)>,
}

#[derive(Deserialize)]
struct Rejected {
    row: usize,
    kind: String,
}

#[derive(Deserialize)]
struct ValidModel {
    file: String,
    rail_abs: f64,
}

#[derive(Deserialize)]
struct StreamCase {
    name: String,
    kind: String,
    bad_row: usize,
    rows: Vec<Vec<Value>>,
}

#[derive(Deserialize)]
struct ValidCase {
    name: String,
    rows: Vec<Vec<Value>>,
}

fn manifest() -> Manifest {
    let text = std::fs::read_to_string(golden("invalid/manifest.json")).unwrap();
    serde_json::from_str(&text).unwrap()
}

/// Numbers stay numbers; the strings "nan", "inf", "-inf" stand for non-finite values.
fn to_f64(v: &Value) -> f64 {
    match v {
        Value::Number(n) => n.as_f64().unwrap(),
        Value::String(s) => s.parse().unwrap(),
        other => panic!("unexpected manifest value {other}"),
    }
}

fn rows(raw: &[Vec<Value>]) -> Vec<Row> {
    raw.iter()
        .map(|r| {
            let mut sample = [0.0; 8];
            for (c, v) in sample.iter_mut().enumerate() {
                *v = to_f64(&r[c]);
            }
            Row {
                sample,
                stim: to_f64(&r[8]),
            }
        })
        .collect()
}

fn model() -> Model {
    Model::from_path(golden("model.json")).unwrap()
}

/// A valid continuation containing an imagery onset (the golden stream's first trial).
fn tail() -> Vec<Row> {
    read_csv(golden("stream.csv")).unwrap()[..2000].to_vec()
}

fn decisions(dec: &mut Decoder, rows: &[Row]) -> Vec<Decision> {
    rows.iter()
        .filter_map(|r| dec.push(&r.sample, r.stim).unwrap())
        .collect()
}

#[test]
fn streaming_rejects_the_bad_row_and_leaves_no_trace() {
    let cases = manifest().streams;
    // bad_row == 0 is what catches latching the offset before validating
    assert!(cases.iter().any(|c| c.bad_row == 0));
    for case in &cases {
        let rows = rows(&case.rows);
        let (good, bad) = (&rows[..case.bad_row], &rows[case.bad_row]);
        let mut dec = Decoder::new(model());
        decisions(&mut dec, good);
        let err = dec.push(&bad.sample, bad.stim).unwrap_err();
        let kind_ok = match (&err, case.kind.as_str()) {
            (CoreError::Stim { index, .. }, "stim")
            | (CoreError::Sample { index, .. }, "sample") => *index == case.bad_row,
            _ => false,
        };
        assert!(kind_ok, "{}: wrong error {err}", case.name);
        assert_eq!(dec.samples_seen(), case.bad_row, "{}", case.name);

        // From here on it must decide exactly like a decoder that never saw the bad row.
        let mut clean = Decoder::new(model());
        decisions(&mut clean, good);
        let after = decisions(&mut dec, &tail());
        assert!(!after.is_empty(), "{}", case.name);
        assert_eq!(after, decisions(&mut clean, &tail()), "{}", case.name);
    }
}

#[test]
fn batch_rejects_the_whole_stream() {
    for case in &manifest().streams {
        let err = decode(&model(), &rows(&case.rows)).unwrap_err();
        let at_bad_row = match (&err, case.kind.as_str()) {
            (CoreError::Stim { index, .. }, "stim")
            | (CoreError::Sample { index, .. }, "sample") => *index == case.bad_row,
            _ => false,
        };
        assert!(at_bad_row, "{}: {err}", case.name);
    }
}

#[test]
fn boundary_codes_are_accepted() {
    let m = manifest();
    assert!(!m.valid_streams.is_empty());
    for case in &m.valid_streams {
        let rows = rows(&case.rows);
        assert!(decode(&model(), &rows).is_ok(), "{}", case.name);
        let mut dec = Decoder::new(model());
        decisions(&mut dec, &rows);
        assert_eq!(dec.samples_seen(), rows.len(), "{}", case.name);
    }
}

#[test]
fn rejected_rows_inside_a_rail_run_leave_the_counters_alone() {
    let m = manifest();
    assert_eq!(m.rail_streams.len(), 3);
    let rail_model = || Model::from_path(golden("rail_model.json")).unwrap();
    for case in &m.rail_streams {
        let expanded: Vec<Vec<Value>> = case
            .rows
            .iter()
            .flat_map(|(n, row)| std::iter::repeat(row.clone()).take(*n))
            .collect();
        let rows = rows(&expanded);
        let mut dec = Decoder::new(rail_model());
        let mut clean = Decoder::new(rail_model());
        let (mut got, mut want) = (Vec::new(), Vec::new());
        for (i, r) in rows.iter().enumerate() {
            if let Some(bad) = case.rejected.iter().find(|b| b.row == i) {
                let seen = dec.samples_seen();
                let err = dec.push(&r.sample, r.stim).unwrap_err();
                let kind_ok = matches!(
                    (&err, bad.kind.as_str()),
                    (CoreError::Stim { .. }, "stim") | (CoreError::Sample { .. }, "sample")
                );
                assert!(kind_ok, "{}: wrong error {err}", case.name);
                assert_eq!(dec.samples_seen(), seen, "{}", case.name);
                continue;
            }
            got.extend(dec.push(&r.sample, r.stim).unwrap());
            want.extend(clean.push(&r.sample, r.stim).unwrap());
        }
        assert_eq!(got, want, "{}", case.name);
        let short: Vec<_> = got
            .iter()
            .map(|d| (d.sample, d.label, d.pinned.clone()))
            .collect();
        assert_eq!(short, case.decisions, "{}", case.name);
    }
}

#[test]
fn every_valid_rail_abs_spelling_loads_and_decodes() {
    let m = manifest();
    assert_eq!(m.valid_models.len(), 3);
    let stream = read_csv(golden("rail_stream.csv")).unwrap();
    let reference = decode(
        &Model::from_path(golden("rail_model.json")).unwrap(),
        &stream,
    )
    .unwrap();
    for case in &m.valid_models {
        let model = Model::from_path(golden(&format!("invalid/{}", case.file))).unwrap();
        assert_eq!(model.rail_abs(), Some(case.rail_abs), "{}", case.file);
        let got = decode(&model, &stream).unwrap();
        if case.rail_abs == 1e6 {
            assert_eq!(got, reference, "{}", case.file);
        } else {
            assert_eq!(got.len(), reference.len(), "{}", case.file);
            assert!(
                got.iter()
                    .all(|d| d.pinned.is_empty() && d.label != mi_core::REFUSED),
                "{}",
                case.file
            );
        }
    }
}
