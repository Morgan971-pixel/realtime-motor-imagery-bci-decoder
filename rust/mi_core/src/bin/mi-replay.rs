//! mi-replay MODEL CSV [--chunk N] [--repeat N] [--paced]
//!
//! Replays a recording through the Rust core and prints one JSON summary line
//! (the format and argument grammar shared with `python -m mi_ref.replay` and the
//! C++ `mi_replay`; see docs/replay.md).

use mi_core::csv::read_csv;
use mi_core::replay::{replay, ReplayOptions, SystemClock};
use mi_core::Model;
use std::io::Write;
use std::process::ExitCode;

const USAGE: &str = "usage: mi-replay MODEL CSV [--chunk N] [--repeat N] [--paced]";

enum Command {
    Help,
    Run(String, String, ReplayOptions),
}

/// A count: ASCII decimal digits, from 1 to 2^64 - 1 (and within usize).
fn count(text: &str) -> Option<usize> {
    if text.is_empty() || !text.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let n: u64 = text.parse().ok()?;
    usize::try_from(n).ok().filter(|&n| n >= 1)
}

fn parse_args(args: &[String]) -> Option<Command> {
    let mut pos = Vec::new();
    let mut opts = ReplayOptions {
        chunk: 25,
        paced: false,
        repeat: 1,
    };
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        // `--chunk=5` is the same as `--chunk 5`
        let (flag, inline) = match arg.split_once('=') {
            Some((f, v)) if f.starts_with("--") => (f, Some(v)),
            _ => (arg.as_str(), None),
        };
        match (flag, inline) {
            ("--chunk", _) => {
                opts.chunk = count(inline.or_else(|| it.next().map(String::as_str))?)?
            }
            ("--repeat", _) => {
                opts.repeat = count(inline.or_else(|| it.next().map(String::as_str))?)?
            }
            ("--paced", None) => opts.paced = true,
            ("-h" | "--help", None) => return Some(Command::Help),
            (f, _) if f.starts_with('-') && f.len() > 1 => return None, // unknown option
            _ => pos.push(arg.clone()),
        }
    }
    let [model, csv]: [String; 2] = pos.try_into().ok()?;
    Some(Command::Run(model, csv, opts))
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (model_path, csv_path, opts) = match parse_args(&args) {
        Some(Command::Run(m, c, o)) => (m, c, o),
        Some(Command::Help) => {
            println!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        None => {
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    let model = match Model::from_path(&model_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("{model_path}: {e}");
            return ExitCode::from(1);
        }
    };
    let rows = match read_csv(&csv_path) {
        Ok(rows) => rows,
        Err(e) => {
            eprintln!("{csv_path}: {e}");
            return ExitCode::from(1);
        }
    };
    if rows.len().checked_mul(opts.repeat).is_none() {
        eprintln!(
            "{USAGE}\n--repeat {} is too large for {} samples",
            opts.repeat,
            rows.len()
        );
        return ExitCode::from(2);
    }
    let r = match replay(&model, &rows, opts, &mut SystemClock::new()) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("{csv_path}: {e}");
            return ExitCode::from(1);
        }
    };
    let decisions: Vec<[u64; 2]> = r
        .decisions
        .iter()
        .map(|d| [d.sample as u64, u64::from(d.label)])
        .collect();
    let summary = serde_json::json!({
        "impl": "rust",
        "chunk": opts.chunk,
        "paced": opts.paced,
        "repeat": opts.repeat,
        "samples": r.samples,
        "elapsed_ns": r.elapsed_ns,
        "decisions": decisions,
        "latencies_ns": r.latencies_ns,
    });
    // a closed pipe (`| head`) is an error to report, not a panic
    if writeln!(std::io::stdout().lock(), "{summary}").is_err() {
        return ExitCode::from(1);
    }
    ExitCode::SUCCESS
}
