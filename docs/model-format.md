# Decoder Contract (schema_version 1)

Every implementation (Python reference, Rust, C++) must follow these steps exactly and in this order. The golden files in `golden/` are the test: same inputs, same outputs, within the tolerances at the bottom.

## Inputs

- `model.json`: produced by `python -m mi_ref.golden` or by training on local data.
- A stream of samples at the model's `fs` (250 Hz for the club headset, 160 Hz for the public EEGMMIDB data): 8 EEG values in channel order `Fz, C3, Cz, C4, Pz, PO7, Oz, PO8`, plus one stim value (0 = no event), validated as in step 4. EEG values must be finite; a NaN or infinite value is an error with the same timing rule as an invalid stim value (step 4).
- Recordings on disk are CSV with the header `timestamps,Fz,C3,Cz,C4,Pz,PO7,Oz,PO8,stim`, one row per sample. The `timestamps` column is not used by the decoder. Dialect: UTF-8 with an optional byte-order mark; LF or CRLF line endings; blank lines (empty, or only spaces and tabs) skipped; a lone CR does not end a line; spaces and tabs around fields and header names allowed (other whitespace characters are outside the dialect, and implementations may treat them differently); exactly 10 comma-separated numbers per row, in decimal or exponent notation (`131`, `131.0`, `.5`, `5.`, `1.31e2`, `1E2`, an optional leading `+` or `-`, and `inf`, `infinity` or `nan` in any letter case, optionally signed; hexadecimal, digit separators such as `1_000`, `nan(...)` payloads, a lone `.` and an exponent with no digits are errors); no quoting and no comments (`#` is an error); `nan` and `inf` parse as numbers and are then rejected by the EEG and stim rules; a number too large for a double reads as infinity (and is rejected the same way), one too small reads as zero. A file must hold at least one sample after the header: a header alone is an error.

## Numeric rules

All arithmetic is IEEE-754 binary64 (`f64` / `double`). No float32, no fast-math, and no fused multiply-add contraction (C++: `-ffp-contract=off`). Variance is two-pass: compute the mean first, then the sum of squared deviations. Parse every number in `model.json` and in the CSV to the correctly rounded double (Rust's `serde_json` needs its `float_roundtrip` feature for this; its default fast path can be one unit in the last place off). With these rules the Rust and C++ cores produce bit-identical decisions. A number written as the integer `-0` is negative zero, the same double as `-0.0`.

## model.json fields

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | int | Must equal 1 |
| `fs` | number > 0 | Sample rate in Hz |
| `channels` | 8 strings | Exactly `["Fz","C3","Cz","C4","Pz","PO7","Oz","PO8"]`, in that order |
| `offset` | string | Exactly `"first_sample"` (step 1) |
| `window.start_s`, `window.len_s` | numbers | Decision window after the imagery onset; `start_s >= 0`, `len_s` must give `L >= 2` samples (step 3), and each of `start_s * fs` and `len_s * fs` must be at most 1048576 (2^20 samples) |
| `bands[i].low_hz`, `bands[i].high_hz` | numbers | Band edges; `0 < low_hz < high_hz < fs / 2` (informational: the filter is `sos`) |
| `bands[i].sos` | k x 6 numbers | Filter sections `[b0, b1, b2, a0, a1, a2]` in order; `a0` must equal 1 and is not used in step 2 |
| `bands[i].csp` | 2p x 8 numbers | Spatial filters, one row per CSP component; every band has the same 2p rows |
| `lda.w` | (bands x 2p) numbers | Weights, band-major: band 0 rows 0..2p-1, then band 1, ... |
| `lda.b` | number | Bias |
| `labels` | object | Exactly `{"score_le_0": 1, "score_gt_0": 2}`: LEFT is 1, RIGHT is 2 |
| `rail_abs` | number > 0, optional | Raw EEG magnitude at or beyond which a channel counts as pinned at the amplifier's limit (step 7). Leave the key out to turn the check off. The only optional key |

**Load errors.** Refuse to load (never fall back) on: anything other than a JSON object where the table expects one (an array in place of `window`, `lda`, `labels`, a band, or the whole document); a wrong `schema_version` (it must be the JSON integer 1), `offset`, `channels` or `labels`; any unknown, missing or repeated key at any level (`rail_abs` is the one key that may be missing); any array with the wrong shape; any numeric field that is not a JSON number (strings such as `"0.5"` and booleans are errors, and JSON has no NaN or Infinity), and any number that overflows a double (`1e400`, a 401-digit integer); `schema_version` and the `labels` values written other than as JSON integers (`1.0` and `1e0` are errors); `rail_abs` present but not a JSON number (`null`, `true` and `"700000"` are errors; leaving the key out is not) or not finite and > 0; `a0 != 1`; a band with no filter sections; a CSP row count that is zero, odd, or different between bands; a band outside `(0, fs/2)`; zero bands; a window shorter than 2 samples, or whose start or length spans more than 2^20 samples (check `seconds * fs` before converting to an integer: it can overflow to infinity).

## Processing steps

1. **Offset.** Remember the first sample of the stream, `x0[c]`. Every sample becomes `x[c] - x0[c]`.
2. **Filter.** For each band and each channel, run the sections in order with zero initial state, using transposed direct form II:

   ```
   cur = input
   for each section s:
       out     = b0*cur + z0[s]
       z0[s]   = b1*cur - a1*out + z1[s]
       z1[s]   = b2*cur - a2*out
       cur     = out
   output = cur
   ```

   State is per (band, channel, section). Never reset it mid-stream.
3. **Window.** `L = round(len_s * fs)` (500). Every seconds-to-samples conversion is `floor(x + 0.5)` for the non-negative `x = seconds * fs`. Write exactly that; do not use Python's half-to-even `round()`. Keep the last `L` filtered samples per band in a ring buffer: sample `i` goes in slot `i % L`, so at sample `i` the oldest sample is in slot `(i + 1) % L`.
4. **Trigger.**
   - **Validation.** A stim value must be finite, a whole number, between 0 and 2147483647 (2^31 - 1), and either 0 or at least 100 (trial >= 1). Anything else is an error, not a skipped event. A batch decoder rejects the whole stream before decoding. A streaming decoder rejects the offending sample before changing any state: the decoder is unchanged and may keep receiving samples, exactly as if that row had never arrived. When a stream has several defects, which one is reported is unspecified; only that the stream (or that sample) is rejected.
   - **Markers.** A nonzero code `c` at sample `i` is a new marker unless sample `i - 1` held the same code. Compare only with the previous sample: `A,A,A` is one marker (at the first `A`), `A,0,A` is two, `A,B,A` is three. The marker's sample is `i`, the first sample of its run.
   - **Decisions.** A marker has `trial = c / 100`, `phase = (c / 10) % 10` and `movement = c % 10` (integer division). Only `phase == 3` with `movement` 1 or 2 schedules a decision, at sample `d = i + round(start_s * fs) + L - 1`. Each marker schedules at most one decision; windows may overlap and each decides independently; two markers can never share a `d`. Decisions come out in order of `d`. If the stream ends before `d`, there is no decision (a decision on the very last sample, `d = n - 1`, does happen).
   - **Output.** Each decision reports `sample` (= `d`), `trial`, `true_label` (= the marker's `movement`), `features`, `score`, `label` (step 6) and `pinned` (step 7).
5. **Features.** At sample `d` (after filtering sample `d`), for each band: `Z = csp @ window` (2p x L, window is channels x time, oldest sample first). Each feature is `ln(max(var(Z_row), 1e-12))` with population variance: subtract the row mean, divide the sum of squares by `L`. The floor means a flat window gives `ln(1e-12)`, never `-inf`. The `max` ignores a NaN variance (IEEE maxNum: Rust `f64::max`, C/C++ `fmax`, NumPy `fmax`), so a window whose values overflowed also gives `ln(1e-12)`; that takes EEG values near 1e308, far outside any real recording, and the cores agree on it.
6. **Score.** Concatenate features band-major (bands x 2p values). `score = sum(w[j] * f[j]) + b`. Label is RIGHT (2) if `score > 0`, else LEFT (1), so a score of exactly 0 is LEFT, unless step 7 refuses the decision.
7. **Pinned channels** (only when `rail_abs` is present). An electrode that loses contact can drive its channel to the amplifier's limit, where it carries no EEG; a model trained with that channel would then decide from noise. The check:
   - **Used channels.** Channel `c` is used if some band's `csp` has a nonzero entry in column `c` (`-0.0` counts as zero). Computed once at load. A model trained without a channel (`python -m mi_ref.evaluate --exclude-pinned`) has a zero column there, so that channel never refuses a decision.
   - **Run counters.** Keep one counter per channel, starting at 0. For every accepted sample (after the input checks of step 4 pass, so a rejected sample changes nothing), compare the raw value, before step 1's offset is removed: if `|x[c]| >= rail_abs` the counter goes up by 1 but never past `L` (`run[c] = min(run[c] + 1, L)`), otherwise it resets to 0. Only `run >= L` matters at a decision, so stopping at `L` changes no decision; it keeps every counter at most 2^20, whatever the integer width. Without `rail_abs` there is no counting.
   - **Decision.** At sample `d`, `pinned` lists, in ascending channel order, the used channels whose counter is at least `L`: every sample of the window sat at or beyond `rail_abs`. If `pinned` is not empty, `label` is 0: the decoder refused because a used channel sat at or beyond `rail_abs` for the whole window. `features` and `score` are still computed and reported. Without `rail_abs`, `pinned` is always empty.
   - **What it does not catch.** A channel that dips below the rail for even one sample of the window decides as normal, however wild its values (`rail_stream` has two such windows). The check refuses only what it can state exactly; it is not a general artifact detector.

## Golden files

`golden/` is the contract. A test checks that the committed files agree with the Python reference within the tolerances below. Regenerate only on purpose (`python -m mi_ref.golden`), review the diff, and commit it with the code change that required it.

| File | Contents |
|---|---|
| `golden/model.json` | Model trained on synthetic recording seed 0 (40 trials) |
| `golden/stream.csv` | Synthetic recording seed 1, 8 trials: 8 decisions |
| `golden/edge_stream.csv` | Short stream built for port traps: 4 s of constant signal holding an imagery onset (every feature hits the variance floor); the same code on two consecutive samples (one marker); phase 1 and 4 markers (ignored); two different imagery codes on adjacent samples (two markers, overlapping windows); an `A,B,A` run (the second `A` is a new marker); an onset whose decision lands exactly on the last sample, and one a sample later that gets none; an imagery onset 100 samples before the end (no decision). 9 decisions |
| `golden/rail_model.json` | `golden/model.json` plus `"rail_abs": 1000000.0`, with CSP columns edited for the used-channel rule: `Oz` is `0.0` in every band and `Pz` is `-0.0` in every band (neither is used: `-0.0` counts as zero); `PO8` is zero in every band but the last (used, through the last band only); `Fz` is zero everywhere but band 2, row 3 (used, through that one weight). Used channels: `[0, 1, 2, 3, 5, 7]` |
| `golden/rail_stream.csv` | `golden/stream.csv` with channels driven to the rail (each decision's window is independent): `C3` at +1e6 from sample 0 through the decision at 1874 (refused, `pinned` `[1]`: its offset-removed value is 0, so only a check on the raw value catches it); `Cz` at +1e6 for exactly the window of 4624 (refused, `[2]`); `Cz` at +1e6 from 50 samples before the window of 7374 through its end except one sample in the middle (decides: the dip resets the run); `C4` at -1e6 for the window of 10124 except its first sample (decides); `PO8`, used only in the last band, for the window of 12874 (refused, `[7]`); `C3` and `PO7` from 50 samples before the window of 15624 through its end (refused, `[1, 5]`); `Pz`, all `-0.0` weights, for the window of 18374 (decides: not used); `Fz`, one nonzero weight, for the window of 21124 (refused, `[0]`); `Oz` at +1e6 throughout (unused, never refuses). Decoded with `rail_model.json`. 8 decisions, 5 refused |
| `golden/expected.json` | For each stream (`stream`, `edge_stream`, `rail_stream`): `band0_filtered_head` (first 1000 filtered samples of band 0, 1000 x 8), `filtered_tail_per_band` (last 100 filtered samples of every band, bands x 100 x 8), `decisions` (one object per decision with `sample`, `trial`, `true_label`, `features`, `score`, `label`, `pinned`). `stream` and `edge_stream` use `model.json` (no `rail_abs`, so every `pinned` is `[]`) |
| `golden/invalid/` | Inputs every implementation must handle, listed in `manifest.json`: `models` (72 invalid model files, eight of them for `rail_abs`, each `{file, reason, rule}`, where `rule` is the kind of load rule broken: `json` not valid JSON (a leading byte-order mark included) or a number no double can hold, `keys` an unknown, missing or repeated key, `type` a wrong JSON type, `shape` a wrong count or length, `value` a forbidden value; `channels` must equal the fixed list, so a list of the wrong length is a `value` error; only the C++ core reports the tag (`mi::ModelError::rule()`); Rust and Python only have to reject, and Python reads an overflowing number as infinity and refuses it as a value), `streams` (13 invalid streams, each `{name, kind, bad_row, rows}`), `valid_streams` (2, each `{name, rows}`), `rail_streams` (3, each `{name, reason, rows, rejected, decisions}`, decoded with `golden/rail_model.json`) and `valid_models` (3, each `{file, reason, rail_abs}`). A row is 9 values: the 8 EEG values in channel order, then stim (no timestamps). An invalid stream is valid up to `bad_row` (which can be 0), which must fail with kind `stim` (a stim rule) or `sample` (a non-finite EEG value) and leave the decoder as if that row never arrived. Every `valid_streams` entry (codes 100 and 2147483647) must be accepted in full. A `rail_streams` entry stores its rows run-length encoded, as `[count, row]` pairs; `rejected` lists `{row, kind}` by index into the expanded rows, and each must fail with that kind and change nothing; `decisions` lists `[sample, label, pinned]` for the accepted rows. Each puts a rejected row inside a run at the rail on `C3`: `L - 1` accepted samples at the rail plus a rejected row at the rail (a bad stim, then a NaN on `Fz`) must decide, and a whole window at the rail with a rejected row below it must refuse, so a decoder that lets a rejected row touch its counters fails. Every `valid_models` file is `golden/rail_model.json` with `rail_abs` written another way (`1000000`, `1e6`, `18446744073709551615`); each must load with the double `rail_abs`, and the first two decode `rail_stream.csv` exactly as `rail_model.json` does. Non-finite values are written as the strings `"nan"`, `"inf"` and `"-inf"` (JSON has no NaN). The fixtures are built from `golden/model.json` and `golden/rail_model.json`: regenerate them (`python -m mi_ref.invalid`) whenever the golden model is regenerated |

## Tolerances

| Quantity | Tolerance |
|---|---|
| Filtered samples | absolute 1e-9 |
| Features | relative 1e-9 |
| Score | absolute 1e-7 |
| Decision sample index, trial, label, pinned, count | exact |
