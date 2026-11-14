//! Bytes become a stream of rows pushed through two decoders: one built from golden/model.json
//! (no rail check) and one from golden/rail_model.json (rail_abs 1e6).
//!
//! Row format, 4 bytes each (scripts/fuzz_seeds.py writes seeds in it): a little-endian i16 base
//! amplitude, a control byte and a rail mask. Channel c gets base * (c + 1) + 1000 * c, or, if
//! bit c of the rail mask is set, rail + |base| with the sign of base, where rail is the rail
//! model's rail_abs (base 0 lands exactly on the rail). The control byte's low 4 bits pick the
//! stim value: 0-9 no event, 10 code 131 (imagery, left), 11 code 132 (imagery, right), 12 code
//! 141 (a valid marker that is not imagery: phase 4), 13 the invalid code 50, 14 NaN, 15 2^31 (too large). If its top bit is set,
//! channel (ctl >> 4) & 7 is then replaced by +infinity.
//!
//! Checks, for each model: push never panics; a rejected row leaves the decoder exactly as if it
//! never arrived (a second decoder fed only the accepted rows makes identical decisions, pinned
//! channels included); an accepted row advances samples_seen by one; every decision's pinned
//! channels are used channels in strictly ascending order (none without rail_abs), and its label
//! is 0 exactly when pinned is not empty, else 2 if score > 0 and 1 otherwise.
#![no_main]

use libfuzzer_sys::fuzz_target;
use mi_core::{Decision, Decoder, Model};
use std::sync::OnceLock;

static MODELS: OnceLock<[Model; 2]> = OnceLock::new();

fn models() -> &'static [Model; 2] {
    MODELS.get_or_init(|| {
        let load = |text: &str| Model::from_json(text).expect("golden model loads");
        [
            load(include_str!("../../../../golden/model.json")),
            load(include_str!("../../../../golden/rail_model.json")),
        ]
    })
}

fn row(chunk: &[u8], rail: f64) -> ([f64; 8], f64) {
    let base = f64::from(i16::from_le_bytes([chunk[0], chunk[1]]));
    let (ctl, mask) = (chunk[2], chunk[3]);
    let mut sample = [0.0; 8];
    for (c, v) in sample.iter_mut().enumerate() {
        *v = if mask & (1 << c) != 0 {
            (rail + base.abs()).copysign(base)
        } else {
            let c = f64::from(c as u8);
            base * (c + 1.0) + 1000.0 * c
        };
    }
    if ctl & 0x80 != 0 {
        sample[usize::from((ctl >> 4) & 7)] = f64::INFINITY;
    }
    let stim = match ctl & 0x0f {
        10 => 131.0,
        11 => 132.0,
        12 => 141.0,
        13 => 50.0,
        14 => f64::NAN,
        15 => 2_147_483_648.0,
        _ => 0.0,
    };
    (sample, stim)
}

fn check(model: &Model, d: &Decision) {
    assert!(d.pinned.windows(2).all(|w| w[0] < w[1]));
    assert!(d.pinned.iter().all(|c| model.used_channels().contains(c)));
    assert!(model.rail_abs().is_some() || d.pinned.is_empty());
    let label = if !d.pinned.is_empty() {
        0
    } else if d.score > 0.0 {
        2
    } else {
        1
    };
    assert_eq!(d.label, label);
}

fuzz_target!(|data: &[u8]| {
    let models = models();
    let rail = models[1].rail_abs().expect("rail_model.json sets rail_abs");
    for model in models {
        let mut all = Decoder::new(model.clone());
        let mut accepted_only = Decoder::new(model.clone());
        for chunk in data.chunks_exact(4) {
            let (sample, stim) = row(chunk, rail);
            let before = all.samples_seen();
            match all.push(&sample, stim) {
                Ok(decision) => {
                    assert_eq!(all.samples_seen(), before + 1);
                    let twin = accepted_only
                        .push(&sample, stim)
                        .expect("an accepted row is accepted again");
                    if let Some(d) = &decision {
                        check(model, d);
                    }
                    assert_eq!(decision, twin);
                }
                Err(_) => assert_eq!(all.samples_seen(), before),
            }
        }
    }
});
