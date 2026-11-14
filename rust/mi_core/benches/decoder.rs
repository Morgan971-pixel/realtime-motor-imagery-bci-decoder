//! Micro-benchmarks: the per-sample cost of `push` and a whole golden stream through `decode`.
//! Run with `cargo bench`; criterion reports the mean with a confidence interval.

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use mi_core::csv::{read_csv, Row};
use mi_core::{decode, Decoder, Model};
use std::hint::black_box;
use std::path::PathBuf;

fn golden(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../golden")
        .join(name)
}

fn benches(c: &mut Criterion) {
    let model = Model::from_path(golden("model.json")).unwrap();
    let rows = read_csv(golden("stream.csv")).unwrap();

    // Markers removed, so no decision ever comes due: this is the cost of filtering and buffering.
    let quiet: Vec<Row> = rows.iter().map(|r| Row { stim: 0.0, ..*r }).collect();
    let mut dec = Decoder::new(model.clone());
    for r in &quiet[..1000] {
        dec.push(&r.sample, r.stim).unwrap();
    }
    let mut i = 0;
    let mut g = c.benchmark_group("push");
    g.throughput(Throughput::Elements(1));
    g.bench_function("sample_without_decision", |b| {
        b.iter(|| {
            let r = &quiet[i % quiet.len()];
            i += 1;
            black_box(dec.push(black_box(&r.sample), r.stim).unwrap())
        })
    });
    g.finish();

    let mut g = c.benchmark_group("decode");
    g.throughput(Throughput::Elements(rows.len() as u64));
    g.sample_size(20);
    g.bench_function("golden_stream", |b| {
        b.iter(|| black_box(decode(&model, black_box(&rows)).unwrap()))
    });
    g.finish();
}

criterion_group!(decoder, benches);
criterion_main!(decoder);
