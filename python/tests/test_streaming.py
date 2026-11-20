import json

import numpy as np
import pytest
from hypothesis import event, given
from hypothesis import strategies as st

from helpers import assert_close_decisions, assert_same_decisions
from mi_ref.golden import CASES, GOLDEN_DIR
from mi_ref.model import Model, decode_recording
from mi_ref.recording import Recording, load_recording
from mi_ref.streaming import StreamingDecoder
from strategies import models, streams


@pytest.fixture(scope="module")
def model():
    return Model.from_json((GOLDEN_DIR / "model.json").read_text())


def _stream(model, rec):
    dec = StreamingDecoder(model)
    return [d for s, c in zip(rec.signal, rec.stim) if (d := dec.push(s, c)) is not None]


@pytest.mark.parametrize("name", CASES)
def test_streaming_matches_committed_golden_decisions(name):
    model = Model.from_json((GOLDEN_DIR / CASES[name]).read_text())
    expected = json.loads((GOLDEN_DIR / "expected.json").read_text())[name]["decisions"]
    assert_same_decisions(_stream(model, load_recording(GOLDEN_DIR / f"{name}.csv")), expected)


@given(st.data())
def test_streaming_matches_batch_on_generated_models_and_streams(data):
    """Any realistic model and stream (extra cues of any code, overlapping windows): streaming
    decisions agree with batch decisions within floating-point noise."""
    model = data.draw(models(realistic=True), label="model")
    signal, stim = data.draw(streams(model, realistic=True), label="stream")
    rec = Recording(np.arange(len(stim)) / model.fs, signal, stim)
    expected = decode_recording(model, rec)
    event("decisions: " + ("0", "1", "2+")[min(len(expected), 2)])  # coverage, see --hypothesis-show-statistics
    assert_close_decisions(_stream(model, rec), expected)


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
