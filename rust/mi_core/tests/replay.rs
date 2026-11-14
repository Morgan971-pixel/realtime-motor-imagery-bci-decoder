//! Replay (docs/replay.md): the same decisions as batch decoding, timed per chunk.

mod common;

use common::*;
use mi_core::csv::read_csv;
use mi_core::replay::{replay, Clock, ReplayOptions, SystemClock};
use mi_core::{decode, Model};
use std::process::Command;

/// Every now_ns() call advances time by 1000 ns; a sleep wakes 0.5 ms late, as on macOS.
#[derive(Default)]
struct FakeClock {
    t: u64,
    sleeps: Vec<u64>,
}

impl Clock for FakeClock {
    fn now_ns(&mut self) -> u64 {
        self.t += 1000;
        self.t
    }
    fn sleep_until_ns(&mut self, target: u64) {
        self.sleeps.push(target);
        self.t = self.t.max(target + 500_000);
    }
}

fn model() -> Model {
    Model::from_path(golden("model.json")).unwrap()
}

fn opts(chunk: usize, paced: bool, repeat: usize) -> ReplayOptions {
    ReplayOptions {
        chunk,
        paced,
        repeat,
    }
}

#[test]
fn replay_makes_the_batch_decisions_for_any_chunk_size() {
    let (model, rows) = (model(), read_csv(golden("edge_stream.csv")).unwrap());
    let want = decode(&model, &rows).unwrap();
    for chunk in [1, 7, 25] {
        let r = replay(
            &model,
            &rows,
            opts(chunk, false, 1),
            &mut SystemClock::new(),
        )
        .unwrap();
        assert_eq!(r.decisions, want, "chunk {chunk}");
        assert_eq!(r.samples, rows.len());
        assert_eq!(r.latencies_ns.len(), r.decisions.len());
        assert!(r.latencies_ns.iter().all(|&x| x <= r.elapsed_ns));
    }
}

#[test]
fn repeat_replays_the_recording_back_to_back() {
    let (model, rows) = (model(), read_csv(golden("edge_stream.csv")).unwrap());
    let twice: Vec<_> = rows.iter().chain(rows.iter()).copied().collect();
    // 9250 % 7 = 3: with chunk 7 a chunk straddles the wrap between passes
    for chunk in [7, 25] {
        let r = replay(
            &model,
            &rows,
            opts(chunk, false, 2),
            &mut SystemClock::new(),
        )
        .unwrap();
        assert_eq!(r.samples, twice.len());
        assert_eq!(
            r.decisions,
            decode(&model, &twice).unwrap(),
            "chunk {chunk}"
        );
    }
}

#[test]
fn fast_mode_times_each_decision_from_its_chunk_handover() {
    let (model, rows) = (model(), read_csv(golden("edge_stream.csv")).unwrap());
    let mut clock = FakeClock::default();
    let r = replay(&model, &rows, opts(25, false, 1), &mut clock).unwrap();
    assert!(clock.sleeps.is_empty());
    assert_eq!(r.latencies_ns, vec![1000; 9]);
}

#[test]
fn paced_mode_waits_for_each_chunk_then_times_from_its_handover() {
    let (model, rows) = (model(), read_csv(golden("edge_stream.csv")).unwrap());
    let total = 2 * rows.len();
    let mut clock = FakeClock::default();
    let r = replay(&model, &rows, opts(7, true, 2), &mut clock).unwrap();
    // chunk k is due when its last sample has arrived: start + (samples so far) / fs; the last
    // chunk is short (18500 % 7 = 6) and one chunk spans the wrap between the two passes
    let ends: Vec<usize> = (7..total).step_by(7).chain([total]).collect();
    let due: Vec<u64> = ends
        .iter()
        .map(|&e| 1000 + (e as f64 * 1e9 / model.fs()) as u64)
        .collect();
    assert_eq!(clock.sleeps, due);
    // one clock reading per decision since the handover: a second decision in the same chunk
    // (samples 2724 and 2726 share one) reads 2000 ns
    let in_chunk: Vec<u64> = (0..r.decisions.len())
        .map(|i| {
            let c = r.decisions[i].sample / 7;
            r.decisions[..=i]
                .iter()
                .filter(|d| d.sample / 7 == c)
                .count() as u64
        })
        .collect();
    assert_eq!(r.decisions.len(), 20);
    assert!(in_chunk.contains(&2));
    assert_eq!(
        r.latencies_ns,
        in_chunk.iter().map(|k| 1000 * k).collect::<Vec<_>>()
    );
    assert!(r.elapsed_ns >= (total as f64 * 1e9 / model.fs()) as u64);
}

#[test]
fn paced_mode_really_runs_at_the_sampling_rate() {
    let model = model();
    let rows = read_csv(golden("edge_stream.csv")).unwrap()[..50].to_vec();
    let r = replay(&model, &rows, opts(25, true, 1), &mut SystemClock::new()).unwrap();
    assert!(
        r.elapsed_ns >= 200_000_000,
        "50 samples at 250 Hz take at least 0.2 s"
    );
}

#[test]
#[should_panic(expected = "must be at least 1")]
fn chunk_must_be_positive() {
    let rows = read_csv(golden("edge_stream.csv")).unwrap();
    let _ = replay(&model(), &rows, opts(0, false, 1), &mut SystemClock::new());
}

#[test]
#[should_panic(expected = "too long to repeat")]
fn repeat_that_overflows_the_sample_count_panics() {
    let rows = read_csv(golden("edge_stream.csv")).unwrap();
    let _ = replay(
        &model(),
        &rows,
        opts(25, false, usize::MAX / rows.len() + 1),
        &mut SystemClock::new(),
    );
}

#[test]
#[should_panic(expected = "must be at least 1")]
fn repeat_must_be_positive() {
    let rows = read_csv(golden("edge_stream.csv")).unwrap();
    let _ = replay(&model(), &rows, opts(25, false, 0), &mut SystemClock::new());
}

fn run_cli(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_mi-replay"))
        .args(args)
        .output()
        .unwrap()
}

#[test]
fn cli_prints_one_json_summary() {
    let (m, csv) = (golden("model.json"), golden("edge_stream.csv"));
    let out = run_cli(&[m.to_str().unwrap(), csv.to_str().unwrap(), "--chunk", "1"]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let s: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(s["impl"], "rust");
    assert_eq!(
        (
            s["chunk"].as_u64(),
            s["paced"].as_bool(),
            s["repeat"].as_u64()
        ),
        (Some(1), Some(false), Some(1))
    );
    assert_eq!(s["samples"], 9250);
    let want: Vec<[u64; 2]> = decode(&model(), &read_csv(&csv).unwrap())
        .unwrap()
        .iter()
        .map(|d| [d.sample as u64, d.label as u64])
        .collect();
    assert_eq!(s["decisions"], serde_json::json!(want));
    assert_eq!(s["latencies_ns"].as_array().unwrap().len(), want.len());
    assert!(s["elapsed_ns"].as_u64().unwrap() > 0);
}

#[test]
fn cli_takes_option_values_after_an_equals_sign_too() {
    let (m, csv) = (golden("model.json"), golden("edge_stream.csv"));
    let (m, csv) = (m.to_str().unwrap(), csv.to_str().unwrap());
    let spaced = run_cli(&[m, csv, "--chunk", "7", "--repeat", "2"]);
    let joined = run_cli(&[m, csv, "--chunk=7", "--repeat=2"]);
    let get = |o: &std::process::Output| {
        serde_json::from_slice::<serde_json::Value>(&o.stdout).unwrap()["decisions"].clone()
    };
    assert_eq!(get(&spaced), get(&joined));
}

#[test]
fn cli_help_prints_usage_and_exits_0() {
    let out = run_cli(&["-h"]);
    assert_eq!(out.status.code(), Some(0));
    assert!(String::from_utf8_lossy(&out.stdout).starts_with("usage: mi-replay"));
}

#[test]
fn cli_rejects_bad_arguments_with_usage() {
    let m = golden("model.json");
    let m = m.to_str().unwrap();
    let huge = "18446744073709551615"; // fits a count, but 9250 x it overflows the sample count
    for args in [
        vec![],
        vec![m],
        vec![m, "x.csv", "--chunk", "0"],
        vec![m, "x.csv", "--chunk", "+5"],
        vec![m, "x.csv", "--chunk"],
        vec![m, "x.csv", "--rep", "2"],
        vec![m, "x.csv", "-x"],
        vec![m, "x.csv", "y.csv"],
        vec![m, "x.csv", "--paced=yes"],
        vec![
            m,
            golden("edge_stream.csv").to_str().unwrap(),
            "--repeat",
            huge,
        ],
    ] {
        let out = run_cli(&args);
        assert_eq!(out.status.code(), Some(2), "{args:?}");
        assert!(String::from_utf8_lossy(&out.stderr).contains("usage: mi-replay"));
    }
}

#[test]
fn cli_reports_a_bad_file_with_its_path() {
    let out = run_cli(&[
        "/nonexistent/model.json",
        golden("edge_stream.csv").to_str().unwrap(),
    ]);
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).starts_with("/nonexistent/model.json: "));
}
