import json
import subprocess
import sys

import numpy as np
import pytest

from mi_ref.golden import GOLDEN_DIR
from mi_ref.model import Model, decode_recording
from mi_ref.recording import Recording, load_recording
from mi_ref.replay import SystemClock, replay
from mi_ref.streaming import StreamingDecoder

EXPECTED = json.loads((GOLDEN_DIR / "expected.json").read_text())


@pytest.fixture(scope="module")
def model():
    return Model.from_json((GOLDEN_DIR / "model.json").read_text())


@pytest.fixture(scope="module")
def edge():
    return load_recording(GOLDEN_DIR / "edge_stream.csv")


class FakeClock:
    """Every now_ns() call advances time by 1000 ns; a sleep wakes 0.5 ms late, as on macOS."""

    def __init__(self):
        self.t = 0
        self.sleeps = []

    def now_ns(self):
        self.t += 1000
        return self.t

    def sleep_until_ns(self, target):
        self.sleeps.append(target)
        self.t = max(self.t, target + 500_000)


def _run(model, rec, clock=None, **opts):
    return replay(StreamingDecoder(model).push, rec.signal, rec.stim, fs=model.fs, clock=clock or SystemClock(), **opts)


def _keys(decisions):
    return [(d["sample"], d["trial"], d["true_label"], d["label"]) for d in decisions]


@pytest.mark.parametrize("chunk", [1, 7, 25])
def test_replay_makes_the_golden_decisions_for_any_chunk_size(model, edge, chunk):
    r = _run(model, edge, chunk=chunk)
    assert _keys(r.decisions) == _keys(EXPECTED["edge_stream"]["decisions"])
    assert r.samples == len(edge.stim)
    assert len(r.latencies_ns) == len(r.decisions)
    assert all(0 <= x <= r.elapsed_ns for x in r.latencies_ns)


@pytest.mark.parametrize("chunk", [7, 25])  # 9250 % 7 = 3: a chunk straddles the wrap between passes
def test_repeat_replays_the_recording_back_to_back(model, edge, chunk):
    r = _run(model, edge, chunk=chunk, repeat=2)
    twice = Recording(np.arange(2 * len(edge.stim)) / model.fs, np.vstack([edge.signal] * 2), np.concatenate([edge.stim] * 2))
    assert r.samples == 2 * len(edge.stim)
    assert _keys(r.decisions) == _keys(decode_recording(model, twice))


def test_fast_mode_times_each_decision_from_its_chunk_handover(model, edge):
    clock = FakeClock()
    r = _run(model, edge, clock=clock, chunk=25)
    assert clock.sleeps == []
    assert r.latencies_ns == [1000] * len(EXPECTED["edge_stream"]["decisions"])


def test_paced_mode_waits_for_each_chunk_then_times_from_its_handover(model, edge):
    clock = FakeClock()
    r = _run(model, edge, clock=clock, chunk=7, paced=True, repeat=2)
    total = 2 * len(edge.stim)
    # chunk k is due when its last sample has arrived: start + (samples so far) / fs; the last
    # chunk is short (18500 % 7 = 6) and one chunk spans the wrap between the two passes
    ends = list(range(7, total, 7)) + [total]
    assert clock.sleeps == [1000 + int(end * 1e9 / model.fs) for end in ends]
    # one clock reading per decision since the handover: a second decision in the same chunk
    # (samples 2724 and 2726 share one) reads 2000 ns
    in_chunk = [sum(e["sample"] // 7 == d["sample"] // 7 for e in r.decisions[: i + 1]) for i, d in enumerate(r.decisions)]
    assert len(r.decisions) == 20 and 2 in in_chunk
    assert r.latencies_ns == [1000 * k for k in in_chunk]
    assert r.elapsed_ns >= int(total * 1e9 / model.fs)


def test_paced_mode_really_runs_at_the_sampling_rate(model, edge):
    short = Recording(edge.timestamps[:50], edge.signal[:50], edge.stim[:50])
    r = _run(model, short, chunk=25, paced=True)  # 50 samples at 250 Hz: at least 0.2 s
    assert r.elapsed_ns >= 200_000_000


@pytest.mark.parametrize("opts", [{"chunk": 0}, {"repeat": 0}])
def test_chunk_and_repeat_must_be_positive(model, edge, opts):
    with pytest.raises(ValueError, match="must be at least 1"):
        _run(model, edge, **opts)


def test_cli_prints_one_json_summary(tmp_path):
    out = subprocess.run(
        [sys.executable, "-m", "mi_ref.replay", str(GOLDEN_DIR / "model.json"), str(GOLDEN_DIR / "edge_stream.csv"), "--chunk", "1"],
        check=True, capture_output=True, text=True,
    ).stdout
    summary = json.loads(out)
    assert summary["impl"] == "python"
    assert (summary["chunk"], summary["paced"], summary["repeat"]) == (1, False, 1)
    assert summary["samples"] == 9250
    assert summary["decisions"] == [[d["sample"], d["label"]] for d in EXPECTED["edge_stream"]["decisions"]]
    assert len(summary["latencies_ns"]) == len(summary["decisions"])
    assert summary["elapsed_ns"] > 0
