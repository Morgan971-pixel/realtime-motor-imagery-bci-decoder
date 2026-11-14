"""Export golden vectors (synthetic only) that the Rust and C++ cores must reproduce.

golden/ is the contract. Regenerate it only on purpose (python -m mi_ref.golden), review the
diff, and commit it together with the code change that required it.
"""
import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np

from .epochs import to_samples
from .filters import filter_causal, remove_offset
from .model import Model, decode_recording, train
from .recording import EEG_CHANNELS, Recording, load_recording, save_recording
from .synth import make_recording

FILTER_CHECK_SAMPLES = 1000
TAIL_SAMPLES = 100
FLAT_SAMPLES = 1000  # the edge stream starts with 4 s of a constant signal
GOLDEN_DIR = Path(__file__).resolve().parents[2] / "golden"
RAIL_ABS = 1e6  # above every raw value of the synthetic streams (offsets reach 8e5)
UNUSED = EEG_CHANNELS.index("Oz")  # the rail model gives this channel no CSP weight (+0.0)
NEG_ZERO = EEG_CHANNELS.index("Pz")  # ... and this one only -0.0 weights: also not used
LAST_BAND_ONLY = EEG_CHANNELS.index("PO8")  # used, but only through the last band
ONE_WEIGHT = EEG_CHANNELS.index("Fz")  # used through a single CSP weight:
ONE_WEIGHT_BAND, ONE_WEIGHT_ROW = 2, 3  # band 2 (of 4), row 3 (its last)
# stream name -> model file it is decoded with
CASES = {"stream": "model.json", "edge_stream": "model.json", "rail_stream": "rail_model.json"}


def make_edge_stream(seed: int = 2) -> Recording:
    """A short stream built to hit the cases a port is most likely to get wrong."""
    rec = make_recording(n_trials=3, seed=seed)
    signal, stim = rec.signal.copy(), rec.stim.copy()
    n = len(stim)
    signal[:FLAT_SAMPLES] = signal[0]  # exactly flat: filtered output is exactly zero
    edits = {
        250: 931,  # imagery onset inside the flat part: every feature hits the variance floor
        251: 931,  # same code on the next sample: the same marker, not a second one
        300: 911,  # phase 1: not imagery, ignored
        320: 941,  # phase 4: not imagery, ignored
        n - 100: 832,  # imagery onset too close to the end for a full window: no decision
        n - 625: 731,  # decision lands exactly on the last sample (d = n - 1)
        n - 624: 732,  # one sample later: the decision would be at d = n, past the end, so none
        2000: 431,  # two DIFFERENT imagery codes on adjacent samples: two markers,
        2001: 432,  # two decisions one sample apart, with overlapping windows
        2100: 531,  # A,B,A: the second 531 repeats 551, not 531, so it is a new marker
        2101: 551,
        2102: 531,
    }
    for i, code in edits.items():
        if stim[i] != 0:
            raise RuntimeError(f"edge marker at {i} would overwrite code {stim[i]}")
        stim[i] = code
    return Recording(timestamps=rec.timestamps, signal=signal, stim=stim)


def make_rail_model(model: Model) -> Model:
    """The golden model plus rail_abs, with CSP columns edited for the used-channel rule:
    Oz all +0.0 and Pz all -0.0 (neither is used: -0.0 counts as zero), PO8 zero in every band
    but the last (used, through the last band only), and Fz zero everywhere but one row of one
    band (used, through that single weight)."""
    csp = [W.copy() for W in model.csp]
    for b, W in enumerate(csp):
        W[:, UNUSED] = 0.0
        W[:, NEG_ZERO] = -0.0
        if b != len(csp) - 1:
            W[:, LAST_BAND_ONLY] = 0.0
        if b != ONE_WEIGHT_BAND:
            W[:, ONE_WEIGHT] = 0.0
        else:
            W[np.arange(len(W)) != ONE_WEIGHT_ROW, ONE_WEIGHT] = 0.0
    return dataclasses.replace(model, csp=csp, rail_abs=RAIL_ABS)


def make_rail_stream(stream: Recording, window_len: int) -> Recording:
    """The golden stream (decisions at 1874 + 2750 k, k = 0..7) with channels driven to the rail.

    Each decision window tests one thing, and the windows do not overlap:
      0. C3 at the rail from sample 0 through the decision (refused): its offset-removed value
         is 0, so a check on that value would miss it.
      1. Cz at the rail for exactly the window (refused).
      2. Cz at the rail from before the window through the decision, except one sample in the
         middle (decides): the dip must reset the run.
      3. C4 at minus the rail for all but the window's first sample (decides).
      4. PO8, used only in the last band, at the rail for the window (refused).
      5. C3 and PO7 at the rail past a whole window (refused, both named).
      6. Pz, whose CSP weights are all -0.0, at the rail for the window (decides: not used).
      7. Fz, used through one CSP weight, at the rail for the window (refused).
    Oz, which is not used, sits at the rail throughout and never refuses.
    """
    L = window_len
    signal = stream.signal.copy()
    fz, c3, cz, c4, pz, po7, po8 = (EEG_CHANNELS.index(n) for n in ("Fz", "C3", "Cz", "C4", "Pz", "PO7", "PO8"))
    d = [1874 + k * 2750 for k in range(8)]
    signal[: d[0] + 1, c3] = RAIL_ABS
    signal[d[1] - L + 1 : d[1] + 1, cz] = RAIL_ABS  # exactly L samples ending at the decision
    signal[d[2] - L - 50 : d[2] + 1, cz] = RAIL_ABS
    signal[d[2] - L // 2, cz] = 0.0  # one sample in the middle of the window breaks the run
    signal[d[3] - L : d[3] + 1, c4] = -RAIL_ABS  # the rail counts both signs ...
    signal[d[3] - L + 1, c4] = 0.0  # ... but one sample inside the window breaks the run
    signal[d[4] - L + 1 : d[4] + 1, po8] = RAIL_ABS
    signal[d[5] - L - 50 : d[5] + 1, [c3, po7]] = RAIL_ABS
    signal[d[6] - L + 1 : d[6] + 1, pz] = RAIL_ABS
    signal[d[7] - L + 1 : d[7] + 1, fz] = RAIL_ABS
    signal[:, UNUSED] = RAIL_ABS
    return Recording(timestamps=stream.timestamps, signal=signal, stim=stream.stim.copy())


def _expected(model: Model, stream: Recording) -> dict:
    x = remove_offset(stream.signal)
    return {
        "band0_filtered_head": filter_causal(model.sos[0], x)[:FILTER_CHECK_SAMPLES].tolist(),
        "filtered_tail_per_band": [filter_causal(sos, x)[-TAIL_SAMPLES:].tolist() for sos in model.sos],
        "decisions": decode_recording(model, stream),
    }


def export_golden(out_dir: Path = GOLDEN_DIR, train_seed: int = 0, stream_seed: int = 1, n_stream_trials: int = 8) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model = train([make_recording(n_trials=40, seed=train_seed)])
    (out_dir / "model.json").write_text(model.to_json())
    save_recording(out_dir / "stream.csv", make_recording(n_trials=n_stream_trials, seed=stream_seed))
    save_recording(out_dir / "edge_stream.csv", make_edge_stream())
    model = Model.from_json((out_dir / "model.json").read_text())
    rail_model = make_rail_model(model)
    (out_dir / "rail_model.json").write_text(rail_model.to_json())
    window_len = to_samples(model.window_len_s, model.fs)
    save_recording(out_dir / "rail_stream.csv", make_rail_stream(load_recording(out_dir / "stream.csv"), window_len))

    # Recompute from the files on disk so goldens match exactly what the native cores read.
    expected = {
        name: _expected(Model.from_json((out_dir / model_file).read_text()), load_recording(out_dir / f"{name}.csv"))
        for name, model_file in CASES.items()
    }
    (out_dir / "expected.json").write_text(json.dumps(expected, indent=1, allow_nan=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=GOLDEN_DIR)
    export_golden(ap.parse_args().out)


if __name__ == "__main__":
    main()
