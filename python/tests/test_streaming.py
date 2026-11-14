import json

import numpy as np
import pytest

from mi_ref.golden import CASES, GOLDEN_DIR
from mi_ref.model import Model, decode_recording
from mi_ref.recording import Recording, load_recording
from mi_ref.streaming import StreamingDecoder
from mi_ref.synth import make_recording


@pytest.fixture(scope="module")
def model():
    return Model.from_json((GOLDEN_DIR / "model.json").read_text())


def _stream(model, rec):
    dec = StreamingDecoder(model)
    return [d for s, c in zip(rec.signal, rec.stim) if (d := dec.push(s, c)) is not None]


def _assert_same(got, expected):
    keys = ("sample", "trial", "true_label", "label", "pinned")
    assert [tuple(d[k] for k in keys) for d in got] == [tuple(e[k] for k in keys) for e in expected]
    for d, e in zip(got, expected):
        np.testing.assert_allclose(d["features"], e["features"], rtol=1e-9)
        assert abs(d["score"] - e["score"]) < 1e-7


@pytest.mark.parametrize("name", CASES)
def test_streaming_matches_committed_golden_decisions(name):
    model = Model.from_json((GOLDEN_DIR / CASES[name]).read_text())
    expected = json.loads((GOLDEN_DIR / "expected.json").read_text())[name]["decisions"]
    _assert_same(_stream(model, load_recording(GOLDEN_DIR / f"{name}.csv")), expected)


@pytest.mark.parametrize("seed", range(5))
def test_streaming_matches_batch_on_random_marker_patterns(model, seed):
    """Random codes, runs, A,B,A patterns and overlapping windows: streaming == batch."""
    rng = np.random.default_rng(seed)
    base = make_recording(n_trials=1, seed=100 + seed)
    stim = np.zeros(len(base.stim), dtype=np.int64)
    for i in rng.choice(len(stim) - 3, size=25, replace=False):
        code = int(rng.choice([131, 132, 121, 151, 111, 141, 232, 331]))
        stim[i : i + int(rng.integers(1, 4))] = code  # runs of 1-3 samples
    rec = Recording(base.timestamps, base.signal, stim)
    _assert_same(_stream(model, rec), decode_recording(model, rec))


@pytest.mark.parametrize("code", [-1, 99, 313.5, 2**31, float("nan")])
def test_bad_code_is_rejected_without_changing_state(model, code):
    dec = StreamingDecoder(model)
    dec.push(np.zeros(8), 0)
    with pytest.raises(ValueError):
        dec.push(np.ones(8), code)
    assert dec._index == 0  # the failed push did not advance the decoder


def test_wrong_sample_shape_is_rejected(model):
    with pytest.raises(ValueError, match="shape"):
        StreamingDecoder(model).push(np.zeros(7), 0)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_sample_is_rejected_without_changing_state(model, bad):
    dec = StreamingDecoder(model)
    dec.push(np.zeros(8), 0)
    sample = np.zeros(8)
    sample[2] = bad
    with pytest.raises(ValueError, match="NaN or infinite"):
        dec.push(sample, 0)
    assert dec._index == 0
