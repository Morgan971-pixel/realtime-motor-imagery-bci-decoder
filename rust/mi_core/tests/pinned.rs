//! The pinned-channel check (`rail_abs`): the same cases as python/tests/test_pinned.py.

mod common;

use common::golden;
use mi_core::{CoreError, Decision, Decoder, Model, REFUSED};
use serde_json::{json, Value};

const L: usize = 500;
const ONSET: usize = 10;
const D: usize = ONSET + 125 + L - 1; // the one decision sample: 634
const N: usize = D + 5;
const RAIL: f64 = 1e6;

fn golden_json() -> Value {
    serde_json::from_str(&std::fs::read_to_string(golden("model.json")).unwrap()).unwrap()
}

/// The golden model with `rail_abs` (None leaves it out) and no CSP weight on `unused`.
fn model(rail: Option<f64>, unused: &[usize]) -> Model {
    let mut v = golden_json();
    if let Some(r) = rail {
        v["rail_abs"] = json!(r);
    }
    for band in v["bands"].as_array_mut().unwrap() {
        for row in band["csp"].as_array_mut().unwrap() {
            for &c in unused {
                row[c] = json!(0.0);
            }
        }
    }
    Model::from_json(&v.to_string()).unwrap()
}

/// A quiet stream with one imagery onset; `edit` drives channels to the rail.
fn run(model: Model, edit: impl Fn(usize, &mut [f64; 8])) -> Decision {
    let mut dec = Decoder::new(model);
    let mut out = Vec::new();
    for i in 0..N {
        let mut s = [5e5 + (i % 7) as f64; 8];
        edit(i, &mut s);
        let stim = if i == ONSET { 131.0 } else { 0.0 };
        out.extend(dec.push(&s, stim).unwrap());
    }
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].sample, D);
    out.pop().unwrap()
}

#[test]
fn without_rail_abs_nothing_is_ever_refused() {
    let d = run(model(None, &[]), |_, s| s[2] = RAIL);
    assert!(d.pinned.is_empty());
    assert_ne!(d.label, REFUSED);
}

#[test]
fn a_used_channel_at_the_rail_for_exactly_the_window_refuses() {
    let d = run(model(Some(RAIL), &[]), |i, s| {
        if (D + 1 - L..=D).contains(&i) {
            s[2] = RAIL;
        }
    });
    assert_eq!((d.label, d.pinned.as_slice()), (REFUSED, &[2][..]));
    assert!(d.score.is_finite() && d.features.len() == 16);
}

#[test]
fn one_sample_short_of_the_window_decides() {
    let d = run(model(Some(RAIL), &[]), |i, s| {
        if (D + 2 - L..=D).contains(&i) {
            s[2] = RAIL;
        }
    });
    assert!(d.pinned.is_empty());
    assert_ne!(d.label, REFUSED);
}

#[test]
fn both_signs_count_and_exactly_rail_abs_is_pinned() {
    let d = run(model(Some(RAIL), &[]), |_, s| {
        s[5] = -RAIL;
        s[3] = RAIL * (1.0 - 1e-12); // just inside: never pinned
    });
    assert_eq!(d.pinned, vec![5]);
}

#[test]
fn an_unused_channel_never_refuses() {
    let m = model(Some(RAIL), &[6]);
    assert_eq!(m.used_channels(), &[0, 1, 2, 3, 4, 5, 7]);
    let d = run(m, |_, s| s[6] = RAIL);
    assert!(d.pinned.is_empty());
}

#[test]
fn several_pinned_channels_come_out_in_channel_order() {
    let d = run(model(Some(RAIL), &[]), |_, s| {
        s[7] = RAIL;
        s[0] = RAIL;
        s[4] = -2.0 * RAIL;
    });
    assert_eq!(d.pinned, vec![0, 4, 7]);
    assert_eq!(d.label, REFUSED);
}

#[test]
fn a_rejected_sample_does_not_touch_the_run_counters() {
    // L - 1 accepted samples at the rail, plus two rejected rows at the rail inside the run (one
    // with NaN on another channel, one with a bad stim): counting either would make L
    let mut dec = Decoder::new(model(Some(RAIL), &[]));
    let mut out = Vec::new();
    for i in 0..N {
        let mut s = [5e5; 8];
        if i > D + 1 - L {
            s[2] = RAIL;
        }
        if i == D - 10 {
            let mut nan = s;
            nan[0] = f64::NAN;
            assert!(matches!(dec.push(&nan, 0.0), Err(CoreError::Sample { .. })));
            assert!(matches!(dec.push(&s, 99.0), Err(CoreError::Stim { .. })));
        }
        let stim = if i == ONSET { 131.0 } else { 0.0 };
        out.extend(dec.push(&s, stim).unwrap());
    }
    assert_eq!(out.len(), 1);
    assert!(out[0].pinned.is_empty());
    assert_ne!(out[0].label, REFUSED);
}

#[test]
fn a_dip_in_the_middle_of_the_window_resets_the_run() {
    let d = run(model(Some(RAIL), &[]), |i, s| {
        if i != D - L / 2 {
            s[3] = -RAIL; // the rail counts both signs
        }
    });
    assert!(d.pinned.is_empty());
    assert_ne!(d.label, REFUSED);
}

#[test]
fn rail_abs_is_optional_and_any_json_number() {
    assert_eq!(model(None, &[]).rail_abs(), None);
    let mut v = golden_json();
    v["rail_abs"] = json!(700000);
    assert_eq!(
        Model::from_json(&v.to_string()).unwrap().rail_abs(),
        Some(700000.0)
    );
}

#[test]
fn bad_rail_abs_is_a_load_error() {
    // the crate's own message is pinned exactly; serde's wording for a wrong type is not
    for (value, needle) in [
        (json!(0.0), Some("rail_abs must be finite and > 0")),
        (json!(-1.0), Some("rail_abs must be finite and > 0")),
        (json!("1e6"), None),
        (Value::Null, None),
        (json!(true), None),
    ] {
        let mut v = golden_json();
        v["rail_abs"] = value;
        match Model::from_json(&v.to_string()) {
            Err(CoreError::Model(msg)) => {
                if let Some(n) = needle {
                    assert!(msg.contains(n), "{msg}");
                }
            }
            other => panic!("accepted: {:?}", other.map(|_| ())),
        }
    }
}
