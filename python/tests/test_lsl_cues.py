"""The live runner with a separate cue stream, end to end through real LSL on this machine:
a Unicorn-like EEG stream (16 channels, no stim) plus the cue markers as their own stream."""
import json
import os
import threading
import uuid

import numpy as np
import pytest

from helpers import assert_same_decisions
from mi_ref.cores import make_decoder
from mi_ref.cues import marker_info
from mi_ref.golden import GOLDEN_DIR
from mi_ref.lsl_live import open_streams, parse_channels, run
from mi_ref.lsl_replay import cue_onsets, publish_split, split_stream_info
from mi_ref.lsl_source import LiveSource
from mi_ref.model import Model
from mi_ref.recording import Recording, load_recording

try:
    import pylsl
except ImportError:
    if os.environ.get("REQUIRE_LSL") == "1":
        raise
    pytest.skip("pylsl is not installed", allow_module_level=True)

MODEL_TEXT = (GOLDEN_DIR / "model.json").read_text()
MODEL = Model.from_json(MODEL_TEXT)
RECORDING = load_recording(GOLDEN_DIR / "stream.csv")
EXPECTED = json.loads((GOLDEN_DIR / "expected.json").read_text())["stream"]["decisions"]
POSITIONS = (3, 0, 9, 1, 12, 2, 5, 14)  # model channel k at stream index POSITIONS[k]


def _streams(positions=POSITIONS):
    tag = uuid.uuid4().hex[:12]
    eeg = pylsl.StreamOutlet(split_stream_info(f"eeg-{tag}", f"eeg-{tag}", MODEL.fs, pylsl, positions))
    cues = pylsl.StreamOutlet(marker_info(f"cues-{tag}", f"cues-{tag}", pylsl))
    inlet, eeg_indices, stim_index, cue_inlet = open_streams(
        pylsl, MODEL.fs, f"eeg-{tag}", "EEG", parse_channels(",".join(map(str, positions))), f"cues-{tag}", 5)
    return eeg, cues, inlet, eeg_indices, stim_index, cue_inlet


class Capture:
    """Wraps a decoder and keeps every decision it makes."""

    def __init__(self, decoder):
        self.decoder, self.decisions = decoder, []

    def push(self, sample, stim):
        decision = self.decoder.push(sample, stim)
        if decision is not None:
            self.decisions.append(decision)
        return decision


def test_split_streams_reproduce_the_golden_decisions_in_every_core(any_core):
    eeg, cues, inlet, eeg_indices, stim_index, cue_inlet = _streams()
    assert sorted(eeg_indices) == sorted(POSITIONS) and stim_index is None
    publisher = threading.Thread(target=publish_split, args=(RECORDING, MODEL), kwargs=dict(
        eeg_outlet=eeg, cue_outlet=cues, pylsl_module=pylsl, eeg_positions=POSITIONS, paced=False, cues_first=True))
    capture, lines, messages = Capture(make_decoder(any_core, MODEL_TEXT)), [], []
    publisher.start()
    count = run(LiveSource(inlet, eeg_indices=eeg_indices, stim_index=stim_index, timeout=2,
                           max_samples=len(RECORDING.stim), pylsl_module=pylsl, cue_inlet=cue_inlet,
                           fs=MODEL.fs, report=messages.append), capture, None, lines.append)
    publisher.join(timeout=10)
    assert count == len(RECORDING.stim)
    assert_same_decisions(capture.decisions, EXPECTED)
    assert len(lines) == 8 and messages == []


def test_paced_cues_sent_after_their_samples_still_land_within_the_lag():
    short = Recording(RECORDING.timestamps[:2000], RECORDING.signal[:2000], RECORDING.stim[:2000])  # 8 s
    eeg, cues, inlet, eeg_indices, stim_index, cue_inlet = _streams()
    publisher = threading.Thread(target=publish_split, args=(short, MODEL), kwargs=dict(
        eeg_outlet=eeg, cue_outlet=cues, pylsl_module=pylsl, eeg_positions=POSITIONS, paced=True))
    capture, messages = Capture(make_decoder("python", MODEL_TEXT)), []
    publisher.start()
    # a generous lag: a busy CI machine may push a cue late
    run(LiveSource(inlet, eeg_indices=eeg_indices, stim_index=stim_index, timeout=2, max_samples=2000,
                   pylsl_module=pylsl, cue_inlet=cue_inlet, cue_lag=2.0, fs=MODEL.fs, report=messages.append),
        capture, None, lambda line: None)
    publisher.join(timeout=15)
    assert_same_decisions(capture.decisions, EXPECTED[:1])  # the first decision falls at sample 1874
    assert messages == []


def _bursty(start, n, rng, burst=10, delay=0.02, stalls=((5000, 0.4), (14000, 0.4))):
    """Receive-time stamps like the club recordings: samples arrive in bursts of `burst`, each
    burst stamped late by up to `delay`, and twice the stream stalls for 0.4 s and catches up."""
    fs = MODEL.fs
    t = np.array([start + ((i // burst) * burst + burst - 1) / fs for i in range(n)])
    t += np.repeat(rng.uniform(0, delay, n // burst + 1), burst)[:n]
    for at, extra in stalls:
        hit = (np.arange(n) >= at) & (np.arange(n) < at + int(extra * fs))
        t[hit] = np.maximum(t[hit], start + (at + int(extra * fs)) / fs + 0.01)
    return np.maximum.accumulate(t)


def test_bursty_timestamps_shift_cues_by_a_steady_few_samples():
    """Measured 2026-11-14: cues land 7 to 10 samples early (28 to 40 ms at 250 Hz). The shift is
    how late the bursty stream stamps its samples; smoothing removes the scatter, not the delay.
    The same shift is in a calibration recording and in live use, so the model absorbs it."""
    n = len(RECORDING.stim)
    onsets = cue_onsets(RECORDING.stim)
    eeg, cues, inlet, eeg_indices, _, cue_inlet = _streams(tuple(range(8)))
    start = pylsl.local_clock()
    stamps = _bursty(start, n, np.random.default_rng(0))
    for index, code in onsets:
        cues.push_sample([code], timestamp=start + index / MODEL.fs)
    row = np.zeros(16)
    for i in range(n):
        row[:8] = RECORDING.signal[i]
        eeg.push_sample(row.tolist(), timestamp=float(stamps[i]))
    source = LiveSource(inlet, eeg_indices=eeg_indices, cue_inlet=cue_inlet, cue_lag=1.0, fs=MODEL.fs, timeout=2,
                        max_samples=n, pylsl_module=pylsl)
    with source:
        placed = [i for i, (_, stim, _) in enumerate(source) if stim]
    merger = source.merger
    errors = [p - i for p, (i, _) in zip(placed, onsets)]
    assert len(placed) == len(onsets) and (merger.late, merger.collisions) == (0, 0)
    assert all(-16 <= e <= 0 for e in errors), errors  # measured -10..-7; margin for a busy CI machine
    assert max(errors) - min(errors) <= 7, errors


class ListInlet:
    """A stand-in inlet that serves fixed chunks, then raises LostError."""

    def __init__(self, chunks):
        self.chunks, self.closed = list(chunks), False

    def open_stream(self, timeout):
        pass

    def pull_chunk(self, timeout, max_samples=1024):
        if not self.chunks:
            raise pylsl.util.LostError("gone")
        return self.chunks.pop(0)

    def close_stream(self):
        self.closed = True


def test_a_lost_stream_releases_held_samples_reports_and_ends_cleanly():
    fs = MODEL.fs
    eeg = ListInlet([([[float(i)] * 8 for i in range(10)], [100 + i / fs for i in range(10)])])
    cue = ListInlet([([[231]], [100 + 4 / fs]), ([], [])])
    seen, messages = [], []

    class Decoder:
        def push(self, sample, stim):
            seen.append((sample[0], stim))

    count = run(LiveSource(eeg, eeg_indices=list(range(8)), timeout=0.01, pylsl_module=pylsl, cue_inlet=cue,
                           cue_lag=1.0, fs=fs, report=messages.append), Decoder(), None, print)
    assert count == 10 and [s for s in seen if s[1]] == [(4.0, 231)]
    assert messages[0] == "LSL EEG stream lost: gone"
    assert eeg.closed and cue.closed


def test_late_and_invalid_cues_are_summarised():
    fs = MODEL.fs
    eeg = ListInlet([([[0.0] * 8 for _ in range(10)], [100 + i / fs for i in range(10)]),
                     ([[0.0] * 8], [100 + 10 / fs])])
    cue = ListInlet([([], []), ([[231], ["x"]], [100 + 2 / fs, 100 + 3 / fs]), ([], [])])
    messages = []

    class Decoder:
        def push(self, sample, stim):
            return None

    run(LiveSource(eeg, eeg_indices=list(range(8)), timeout=0.01, pylsl_module=pylsl, cue_inlet=cue,
                   cue_lag=0.0, fs=fs, report=messages.append), Decoder(), None, print)
    assert "dropped late cue 231" in messages[0]
    assert messages[1] == "dropped invalid cue 'x' at 100.012000 s"
    assert messages[-1] == "cues dropped: 1 late, 1 invalid, 0 on an already-cued sample, 0 after the last EEG sample"


@pytest.mark.parametrize("kwargs,message", [({"stim_index": 8, "fs": 250.0}, "its own stim channel"),
                                            ({"fs": None}, "fs is required")])
def test_live_source_refuses_an_unusable_cue_setup(kwargs, message):
    with pytest.raises(ValueError, match=message):
        LiveSource(ListInlet([]), eeg_indices=list(range(8)), pylsl_module=pylsl, cue_inlet=ListInlet([]), **kwargs)


def test_losing_the_cue_stream_keeps_decoding_the_eeg():
    fs = MODEL.fs
    eeg = ListInlet([([[0.0] * 8 for _ in range(5)], [100 + i / fs for i in range(5)]),
                     ([[0.0] * 8 for _ in range(5)], [100 + (5 + i) / fs for i in range(5)])])
    cue = ListInlet([])  # lost from the first pull
    messages = []

    class Decoder:
        def push(self, sample, stim):
            return None

    count = run(LiveSource(eeg, eeg_indices=list(range(8)), timeout=0.01, pylsl_module=pylsl, cue_inlet=cue,
                           cue_lag=0.0, fs=fs, report=messages.append), Decoder(), None, print)
    assert count == 10
    assert messages[0] == "LSL cue stream lost: gone; decoding continues without new cues"
    assert messages[1] == "LSL EEG stream lost: gone"
