# Continuous integration, fuzzing and Miri

`.github/workflows/ci.yml` runs every check this repo has on GitHub's machines. Standard GitHub-hosted runners are free for public repositories, so the public repo runs both jobs on every push and pull request. A private copy spends minutes from the account's monthly Actions quota (2,000 on GitHub Free, 3,000 on Pro), with macOS minutes counted 10 times, so there the macOS job runs only weekly and on demand. The workflow tells the two apart by `github.event.repository.private`.

## When each job runs

| Job | Runs on | When |
|---|---|---|
| Ubuntu | `ubuntu-24.04`, clang | every push to `main`, every pull request, weekly, on demand |
| macOS | `macos-26` (Apple silicon), Apple clang 17 (Xcode 26) | public repo: with Ubuntu, plus weekly and on demand; private copy: weekly (Mondays 06:00 UTC) and on demand only (Actions tab, `ci`, Run workflow) |

Every push to `main` gets a full run, and a push never cancels the weekly run (or the reverse). Only a pull request's run is cancelled when a newer push lands on its branch. Pushes to other branches do not run CI: open a pull request, or press Run workflow on the branch, to test it before merging.

## What the Ubuntu job checks, in order

1. **Data guard.** `scripts/test_hooks.sh` proves the pre-commit hook still blocks data. `scripts/check_tracked.sh` fails if any tracked file matches the data policy in `.gitignore`, ignoring letter case. That catches files that skipped the hook (`git commit --no-verify`, `git add -f`).
2. **Rust.** `cargo fmt --check`, clippy with warnings as errors, and `cargo test`. The fuzz crate gets fmt and clippy too. Then the release `mi-replay` build the CLI tests use.
3. **C++.** `ctest` under the `dev`, `ubsan`, `asan` and `release` presets. ASan (AddressSanitizer, which catches out-of-bounds reads and use-after-free) runs only here: it hangs at startup on macOS.
4. **Python.** The full suite with both native cores built and required (`REQUIRE_CORES=mi_core,mi_core_cpp`), so a core that fails to build fails the run instead of being skipped. Before it, `python -m mi_ref.eegmmidb --subjects 1 --keep-edf` fetches public EEGMMIDB person 1 (about 8 MB; the EDF files are cached under a key built from the pinned checksums) and `REQUIRE_PUBLIC_DATA=1` makes the real-data parity test (docs/public-data.md) mandatory.
5. **Miri.** The Rust library's unit tests under Miri, an interpreter that checks every memory access and reports undefined behaviour. See below for what runs.
6. **Fuzz smoke.** Every fuzz target for 60 seconds, seeded from `golden/`. Each input gets at most 10 seconds (`-timeout=10`), so a hang fails fast instead of hitting the job timeout. If a target finds a crash or a hang, the run fails and the input that caused it is saved as the `fuzz-artifacts` workflow artifact for 14 days.

The macOS job runs steps 1 to 4 with Apple clang, with five differences: no fmt or clippy (only `cargo test` and the release build), no ASan, no fuzzing or Miri, no public data (the real-data parity test skips), and no pylsl (the LSL tests skip). It uses the `macos-26` image because its default Xcode 26 has `std::from_chars` for floating point; the first run, on `macos-15` (Xcode 16.4), failed to compile `cpp/src/csv.cpp` without it. On the public repository it runs with every push and pull request; in a private copy only on the schedule or by hand.

## Toolchains

- Rust stable is pinned to 1.94.1 (`RUST_STABLE` in `ci.yml`), so a new clippy lint cannot fail a run on its own.
- Miri and cargo-fuzz need nightly Rust. CI pins `nightly-2026-11-09` (`RUST_NIGHTLY`), the nightly this setup was verified on.
- To move either one: change the variable, run the same commands locally on the new toolchain, and fix what they report in the same commit.
- Python is 3.11.
- clang 18 builds the C++ on Ubuntu, including libFuzzer, which comes with `libclang-rt-18-dev`.

## Fuzzing

A fuzzer feeds a program millions of machine-mutated inputs and watches for crashes, hangs and sanitizer reports. Each target below accepts any bytes. A rejected input is fine; a panic, an abort, an uncaught exception of the wrong type, or a sanitizer report is a bug.

| Target | Core | Input | Extra check |
|---|---|---|---|
| `model_json` | Rust | bytes as model JSON | an accepted model has `2 <= L <= 2^20`, `start <= 2^20`, at least one band, a `rail_abs` (when present) that is finite and > 0, and used channels in strictly ascending order |
| `parse_csv` | Rust | bytes as a recording CSV | an accepted file has at least one sample |
| `decode_stream` | Rust | bytes as rows pushed through two decoders, built from `golden/model.json` and `golden/rail_model.json` | a rejected row leaves the decoder exactly as if it never arrived (a second decoder fed only the accepted rows makes identical decisions, pinned channels included); `pinned` lists used channels in ascending order (none without `rail_abs`); the label is 0 exactly when `pinned` is not empty |
| `fuzz_model` | C++ | bytes as model JSON | only `mi::ModelError` may be thrown; same checks as `model_json` |
| `fuzz_csv` | C++ | bytes as a recording CSV | only `mi::CsvError` may be thrown; at least one sample |

`decode_stream` reads 4 bytes per row: an i16 amplitude, a control byte that picks the stim value or plants an infinite EEG value, and a mask of channels to drive to the rail. Its source file documents the format.

**Seeds** come from the synthetic `golden/` files only, never from recordings:

    python3 scripts/fuzz_seeds.py fuzz-corpus    # model/, csv/, stream/

`fuzz-corpus/` and `fuzz-artifacts/` are git-ignored. Never commit them: the fuzzers add their own inputs there.

**Run the Rust targets locally** (nightly Rust and `cargo install cargo-fuzz`):

    cd rust/mi_core
    cargo +nightly fuzz run model_json "$PWD/../../fuzz-corpus/model" -- -max_total_time=60
    cargo +nightly fuzz run parse_csv "$PWD/../../fuzz-corpus/csv" -- -max_total_time=60
    cargo +nightly fuzz run decode_stream "$PWD/../../fuzz-corpus/stream" -- -max_total_time=60

**Run the C++ targets locally.** Apple clang has no libFuzzer, so on a Mac the `fuzz-seeds` preset builds the same targets with a plain `main` that runs each file it is given, under UBSan. Replaying the corpus the Rust fuzzers grew is a useful cross-check:

    cd cpp
    cmake --preset fuzz-seeds && cmake --build --preset fuzz-seeds
    find ../fuzz-corpus/model -type f -exec build/fuzz-seeds/fuzz_model {} +
    find ../fuzz-corpus/csv -type f -exec build/fuzz-seeds/fuzz_csv {} +

On Linux with clang, the `fuzz` preset builds real libFuzzer binaries (ASan and UBSan on):

    cmake --preset fuzz && cmake --build --preset fuzz
    build/fuzz/fuzz_model -max_total_time=60 ../fuzz-corpus/model

**When a fuzzer finds a crash:**
1. Download the `fuzz-artifacts` artifact from the failed run.
2. Reproduce it: `cargo +nightly fuzz run <target> <file>` for Rust; for C++, `build/fuzz/fuzz_<target> <file>` on Linux with the `fuzz` preset (ASan and UBSan). The Mac's `build/fuzz-seeds/fuzz_<target> <file>` has UBSan only, so it will not reproduce an AddressSanitizer report.
3. Turn the input into a regular unit test.
4. Fix the bug, keeping the three cores in agreement (docs/model-format.md).

Crash inputs are fuzzer-made bytes, but keep them out of git like everything else in `fuzz-artifacts/`.

## Miri

    cd rust/mi_core
    MIRIFLAGS="-Zmiri-deterministic-floats -Zmiri-disable-isolation" cargo +nightly miri test --lib

- **What runs:** every library unit test: CSV parsing, filter, features, ring buffer, stim rules, timing, the model loader and the pinned-channel run counters. About 25 seconds.
- **What is skipped:** the integration tests in `tests/`, which push tens of thousands of samples, far too slow under an interpreter.
- **Why `-Zmiri-disable-isolation`:** two unit tests read the golden model files from disk, and Miri blocks file access by default. The tests read only files in this repo.
- **Why `-Zmiri-deterministic-floats`:** Miri deliberately perturbs the last bits of `ln()` and other library maths. Real C libraries are allowed to differ there, and Miri wants code to tolerate that. This decoder relies on the platform's `ln` instead, so its exact-match tests would fail, which tells us nothing about memory safety. `-Zmiri-deterministic-floats` turns the perturbation off.
- **Limits:** the core has no `unsafe` code. Miri here mainly guards the dependencies and any `unsafe` added later.

## What a run costs

Nothing on the public repository: standard GitHub-hosted runners are free for public repositories. The numbers below matter only for a private copy.

Measured on GitHub's standard private-repo runners (2-core Linux, 3-core macOS) over the first runs in November 2026. A warm cache saves a few minutes on the Rust steps.

| Job | Wall time | Quota minutes |
|---|---|---|
| Ubuntu | 16 to 23 min measured (5 of them fuzzing) | 16 to 23 per push to `main` or pull request update |
| macOS | about 5 min measured | about 50 per run (10x), about 220 a month at one run a week |

On GitHub Free that leaves room for roughly 80 Ubuntu runs a month next to the weekly macOS run. If the quota gets tight, run the fuzz smoke step only on the weekly schedule: it is the largest single step.
