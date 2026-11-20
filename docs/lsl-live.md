# Live LSL inlet

The live runner uses Python and `pylsl` for Lab Streaming Layer transport. Select the decoder core with `--core python`, `--core rust`, or `--core cpp`; the Rust and C++ options use the existing `mi_core` and `mi_core_cpp` Python bindings. The runner reads EEG samples from LSL chunks, validates them, and can print decisions and publish them as a separate LSL marker stream when stim codes are present on the EEG inlet.

## Install

From the repository root:

```bash
.venv/bin/pip install -e 'python[live]'
```

Build either native binding with the commands in `AGENTS.md` if it is not already available. The Python core does not require an LSL installation unless the live or replay commands are used.

## Synthetic demo

Open two terminals in the repository root. Start the synthetic outlet:

```bash
.venv/bin/python -m mi_ref.lsl_replay golden/model.json golden/stream.csv --name Unicorn
```

Then run the inlet:

```bash
.venv/bin/python -m mi_ref.lsl_live golden/model.json --core python --stream-name Unicorn --stream-type EEG
```

Use `--core rust` or `--core cpp` to select a native core. The replay outlet is paced at the model's `fs` by default; `--unpaced` publishes the same synthetic samples as quickly as possible. Its EEG and stim channels are transmitted as LSL `double64`, preserving the CSV values exactly. It reads only the committed synthetic `golden/stream.csv` in this example.

## Input stream

The live inlet resolves an exact LSL stream name and type (default `Unicorn` and `EEG`). Its channel metadata must name all eight model channels: `Fz`, `C3`, `Cz`, `C4`, `Pz`, `PO7`, `Oz`, and `PO8`. The inlet reorders values by these labels. A single optional channel named `stim` or `Markers` supplies the stim code; without one, every sample has stim 0. Duplicate, missing, or unknown labels are rejected. A nominal rate must match `model.fs` within 1e-6 relative tolerance. The inlet does not infer event codes, resample, or interpolate. LSL receive timestamps do not change decoder timing; the decoder counts accepted samples according to `docs/model-format.md`.

**Other layouts (`--channels`).** A headset stream such as the Unicorn's carries more than the 8 EEG channels (motion, battery, counter) and may label them differently. `--channels` picks the 8 EEG channels and ignores the rest, either by zero-based index in model order (`--channels 0-7`, or `--channels 3,0,1,2,4,5,6,7`) or by label (`--channels "Fz=EEG 1,C3=EEG 2,Cz=EEG 3,C4=EEG 4,Pz=EEG 5,PO7=EEG 6,Oz=EEG 7,PO8=EEG 8"`). Check the real labels in the headset app before trusting either form: the order of the Unicorn's electrodes in its stream is not confirmed here.

**Cues from a separate stream (`--cue-stream`).** The Unicorn stream carries no cues, so without one every sample has stim 0 and nothing is ever decided. `--cue-stream NAME` opens a second inlet on an LSL `Markers` stream (such as `mi-realtime-cues` from `python -m mi_ref.cues`) and gives each cue code to the EEG sample nearest to it in time, exactly as if it had been in a stim column (`python/mi_ref/cue_merge.py`). Markers may be integers, whole-number floats such as `231.0`, or ASCII digit strings of at most 10 characters (surrounding spaces allowed); anything else, or a value outside 100 to 2^31 - 1, is dropped and reported as invalid. If two cues land on the same sample, the cue with the earlier time wins (equal times: the one that arrived first), and the other is dropped and reported. EEG is held back `--cue-lag` seconds (default 0.5) so a cue that arrives slightly after its sample still lands; a cue later than that is dropped and reported, never moved. Cues still waiting when the EEG stream ends are dropped too, and at the end the runner prints how many cues were dropped and why. Both inlets use LSL clock synchronisation, and the EEG inlet also smooths timestamp jitter; neither reconnects silently. If the cue stream is lost (the cue program finished or was restarted), the runner says so and keeps decoding without new cues; if the EEG stream is lost, the run ends with a message. The decoder and the recorder (`mi_ref.lsl_record`) read both streams through the same class (`python/mi_ref/lsl_source.py`), so a recorded session carries exactly the cues the decoder would have seen. Start the decoder a few seconds before the first cue, so LSL's clock synchronisation has settled; the cue program's default 3 s lead-in gives it that time.

For a different stream, pass `--stream-name NAME --stream-type TYPE`. Discovery waits up to 10 seconds by default; set `--timeout SECONDS` to change the limit. The error lists available stream names, types, and channel counts. If the source omits channel labels, the inlet accepts only exactly eight channels and assumes they are already in model order.

## Headset-style demo (no headset)

`--split` makes the replay look like the headset plus the cue program: a 16-channel EEG stream with no stim (EEG at indices 0 to 7, then motion, battery and counter channels) and the stim codes as markers on a separate stream.

```bash
# terminal 1: the decoder, started first so it is listening
.venv/bin/python -m mi_ref.lsl_live golden/model.json --core rust --channels 0-7 --cue-stream mi-realtime-cues --timeout 60
# terminal 2: the fake headset and cue program
.venv/bin/python -m mi_ref.lsl_replay golden/model.json golden/stream.csv --name Unicorn --split mi-realtime-cues
```

A decision appears about every 11 seconds. Sample numbers count from when the decoder joined the stream, so they run a little below the golden ones.

## Cue timing

A cue lands on the EEG sample whose timestamp is nearest to the cue's. Headset streams often stamp samples when they *arrive*, in bursts; the club recordings show gaps from 0 to 0.4 s. Measured on 2026-11-14 with simulated bursts of 10 samples, up to 20 ms extra delay and two 0.4 s stalls (`tests/test_lsl_cues.py`): without smoothing, cues landed 0 or 10 samples early; with the smoothing `open_streams` turns on, 7 to 10 samples (28 to 40 ms) early, every time. The steady part is how late the stream stamps its samples, which no receiver can know. It does no harm here: a calibration recording made through the same merge carries the same shift, so the model trained on it expects it, and it is small next to the decision window, which starts 125 samples after the cue and lasts 500.

## Calibration and live use with the headset

Record only with the club's OK and the wearer's consent. Recordings and trained models are written outside the repository (`mi_ref.lsl_record` and `mi_ref.train` refuse a path inside it, however it is spelled) and never enter git, and neither command overwrites an existing file.

1. Start the Unicorn LSL app on the Windows machine and note its stream name, channel labels and rate. Connect the Mac to the same network.
2. Start the recorder for a bit longer than the session (40 trials at the default timing take 443 s). It waits up to 60 s for the cue stream:

   ```bash
   .venv/bin/python -m mi_ref.lsl_record --out ~/mi-recordings/calib-01.csv --seconds 460 --channels 0-7 --timeout 60
   ```

3. Within that minute, run the cues in a second terminal; the wearer follows them:

   ```bash
   .venv/bin/python -m mi_ref.cues --trials 40
   ```

4. Train and read the honest accuracy:

   ```bash
   .venv/bin/python -m mi_ref.train ~/mi-recordings/calib-01.csv --out ~/mi-recordings/model-01.json
   ```

   Below chance_95, live decisions will be close to guesses: record more, or check the electrodes first.

5. Run live: start the decoder, then, within a minute, a fresh cue session in the other terminal:

   ```bash
   .venv/bin/python -m mi_ref.lsl_live ~/mi-recordings/model-01.json --core rust --channels 0-7 --cue-stream mi-realtime-cues --timeout 60
   .venv/bin/python -m mi_ref.cues --trials 20 --seed 1
   ```

## Decision output

The terminal prints one line per decision with sample index, trial, true label, decoded label, score, and pinned channel names. Label 0 means the pinned-channel check refused the decision. Raw EEG samples are not printed.

The default separate LSL outlet is named `mi-realtime-decisions`, has type `Markers`, and publishes one string channel containing compact JSON with exactly these fields:

```json
{"sample":1874,"trial":1,"true_label":1,"label":2,"score":0.125,"pinned":[]}
```

Other LSL programs can resolve the marker stream by name and type, then subscribe to its string samples. Change its name and source ID with `--marker-name` and `--marker-source-id`. `--no-marker-outlet` disables publishing for diagnosis.

## Troubleshooting

- **No matching stream:** verify the exact name and type in the LSL app, pass them explicitly, and increase `--timeout`.
- **Stream is visible but connection fails:** confirm the machines share a network that permits LSL traffic and that host firewalls allow the LSL ports. The Windows Unicorn app and Mac must be on the same reachable club network.
- **Channel layout rejected:** without `--channels`, the runner accepts exactly the eight model labels and at most one `stim`/`Markers` channel. For the Unicorn, pass `--channels`.
- **No decisions:** the decoder only decides after an imagery cue. Pass `--cue-stream` and start the cue program, or use a stream with a stim channel.
- **Cues reported late:** raise `--cue-lag` (for example to 1.0); decisions then come out that much later.
- **Sample rate rejected:** use a model trained for the stream's nominal rate. This phase intentionally does not resample live data.
- **Native core unavailable:** use `--core python`, or build the Rust/C++ Python binding in `.venv` using the repository setup instructions.
- **pylsl/liblsl load error:** reinstall with `.venv/bin/pip install -e 'python[live]'`; consult pylsl's install notes if liblsl cannot load on the machine.

## Manual club check

Pending club access; automated tests use only fake streams.

1. Confirm the Unicorn stream's name, type, rate (250 Hz) and channel labels; pick the `--channels` form that matches them.
2. With the club's OK and the wearer's consent, record a calibration session, train, and note the cross-validated accuracy against chance.
3. Run live with the cue program; check that every imagery cue produces one decision and that no cues are reported late or dropped.
4. Keep nothing in the repository; delete recordings the club does not want kept.
