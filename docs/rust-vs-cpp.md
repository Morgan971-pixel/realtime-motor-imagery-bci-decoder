# Rust vs C++: one decoder, written twice

**Short version:** both cores implement the same written contract (`docs/model-format.md`) and make bit-identical decisions, at the same speed (about 18 µs per decision, median). Neither core has had a memory-safety bug. The one memory bug in the project was in C++ *test* code, found by AddressSanitizer on Linux CI; Rust's compiler rejects that pattern outright. Most real differences between the cores came from libraries and toolchains (how each parses a number, what each JSON library accepts), and the bit-for-bit comparison between the cores is what caught them.

## Same spec, two codebases

| | Rust (`rust/mi_core`) | C++20 (`cpp/`) |
|---|---|---|
| Core library, non-blank lines | 1,074 in 11 files | 897 in 20 files (9 `.cpp`, 11 headers) |
| Python bindings | 76 (PyO3) | 85 (pybind11) |
| Replay command | 116 | 127 |
| Own tests | 59 (unit, golden, invalid-input, property-based) | 57, run under 4 build presets |
| `unsafe` / sanitizer-only checks | 0 `unsafe` blocks; Miri on 23 unit tests | UBSan + library bounds checks everywhere, ASan on Linux |
| Fuzz targets | 3 (cargo-fuzz) | 2 (libFuzzer) |

Both pass the same fixtures: the synthetic golden recordings and 72 invalid models plus 13 invalid streams in `golden/invalid/`. The Python suite (792 tests) checks each core against the Python reference through its bindings, on synthetic data and on real public EEG.

**Speed** (`python -m mi_ref.bench`, Apple M4, from the README table):

| | Rust | C++ | Python reference |
|---|---:|---:|---:|
| Median latency per decision | 18.1 µs | 18.5 µs | 193.9 µs |
| Samples per second | 6.8 M | 7.6 M | 15 k |
| Peak memory | 7.7 MiB | 11.7 MiB | 108 MiB |

The two native cores are within noise of each other on latency, which is what you'd expect from the same arithmetic in the same order compiled to similar machine code.

## What each side's tooling caught

**Found by comparing the cores bit for bit (neither compiler could have):**
- **Number parsing.** `serde_json`'s default float parser is fast but not always correctly rounded. Model weights loaded in Rust differed from C++ in their last bits, so the cores' features did too. Turning on `serde_json`'s `float_roundtrip` feature made them identical.
- **NaN variance.** The Rust core floored a window's variance with `fmax`, which turns NaN into the floor; C++ and Python passed NaN through. All three now use `fmax` (`cpp/src/features.cpp`).
- **A byte-order mark.** A `model.json` starting with a UTF-8 BOM loaded in C++ (nlohmann/json skips it) but not in Rust. The contract now refuses it, and a fixture tests both.
- **Repeated JSON keys.** nlohmann/json silently keeps the last of two identical keys; serde's derive rejects them. C++ needed its own check (`cpp/src/json_strict.cpp`, 81 lines) to match.

**Found by C++ sanitizers and toolchains:**
- **Use-after-free in a test (AddressSanitizer, Linux CI).** A range-for loop over a member of a temporary JSON object read freed memory. In C++20 the temporary dies before the loop body runs. It passed in debug, UBSan and release builds on macOS because the freed bytes still held the old values. Details in the README's lessons learned. Rust's borrow checker refuses to compile the equivalent code.
- **`std::from_chars` out of range.** On an out-of-range number the standard leaves the output unspecified; libc++ fills it in anyway, other libraries may not. `cpp/src/csv.cpp` now decides overflow or underflow from the text itself.
- **Missing library support.** Xcode 16's libc++ has no floating-point `std::from_chars`, so the C++ core needs Xcode 26 on macOS (CI moved to the `macos-26` image).
- **ASan on macOS** hangs at startup on this machine, so ASan runs only on Linux CI; locally, UBSan plus the standard library's bounds checks is the gate.

**Found by Rust tooling:**
- No bug in the Rust core was found by Miri, fuzzing or the property tests. They ran clean, which is evidence, not proof.
- Friction rather than bugs: `cargo test --features python` fails to link on macOS (the PyO3 extension-module setting leaves libpython unlinked), so the bindings are tested from Python. Miri also needs `-Zmiri-deterministic-floats`, because by default it perturbs the last bits of `ln()` and breaks the exact-match tests.

**Fuzzing:** every target runs for 60 seconds on every CI run. No fuzzer has reported a crash so far.

## What I'd take from it

- **For bit-exact agreement, the language mattered less than the libraries.** Number parsing and JSON edge cases caused every cross-core difference. A written contract plus shared fixtures turned each one into a failing test instead of a silent drift.
- **Rust moved memory safety to compile time; C++ needed sanitizers and a second operating system.** The one memory bug lived in code that passed every macOS build. Without ASan on Linux it would still be there.
- **Neither core was faster in a way that matters.** About 18 µs per decision is roughly 0.5% of the 4 ms between samples at 250 Hz, so the choice between them comes down to tooling and team, not speed.
