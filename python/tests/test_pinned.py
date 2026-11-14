"""The pinned-channel check: a decision refuses (label 0) when a used channel sat at or beyond
rail_abs for its whole window. Streaming and batch decoding must agree on every case."""
import dataclasses
import json

import numpy as np
import pytest

from mi_ref.golden import GOLDEN_DIR
from mi_ref.model import REFUSED, Model, decode_recording
from mi_ref.recording import Recording
from mi_ref.streaming import StreamingDecoder

BASE = Model.from_json((GOLDEN_DIR / "model.json").read_text())
L, START = 500, 125
ONSET = 10
D = ONSET + START + L - 1  # the one decision sample: 634
N = D + 5
RAIL = 1e6


def _model(rail=RAIL, unused=()):
    csp = [W.copy() for W in BASE.csp]
    for W in csp:
        W[:, list(unused)] = 0.0
    return dataclasses.replace(BASE, csp=csp, rail_abs=rail)


def _recording(edit=None):
    rng = np.random.default_rng(0)
    signal = 5e5 + rng.normal(0.0, 5.0, size=(N, 8))
    if edit:
        edit(signal)
    stim = np.zeros(N, dtype=np.int64)
    stim[ONSET] = 131
    return Recording(np.arange(N) / 250.0, signal, stim)


def _both(model, rec):
    """Batch and streaming decisions, checked equal, returned once."""
    batch = decode_recording(model, rec)
    dec = StreamingDecoder(model)
    streamed = [d for s, c in zip(rec.signal, rec.stim) if (d := dec.push(s, c)) is not None]
    assert [(d["sample"], d["label"], d["pinned"]) for d in streamed] == [(d["sample"], d["label"], d["pinned"]) for d in batch]
    np.testing.assert_allclose([d["score"] for d in streamed], [d["score"] for d in batch], rtol=0, atol=1e-7)
    assert len(batch) == 1 and batch[0]["sample"] == D
    return batch[0]


def _at_rail(channel, first, last, value=RAIL):
    def edit(signal):
        signal[first : last + 1, channel] = value
    return edit


def test_without_rail_abs_nothing_is_ever_refused():
    d = _both(dataclasses.replace(BASE, rail_abs=None), _recording(_at_rail(2, 0, N - 1)))
    assert d["pinned"] == [] and d["label"] != REFUSED


def test_a_used_channel_at_the_rail_for_exactly_the_window_refuses():
    d = _both(_model(), _recording(_at_rail(2, D - L + 1, D)))
    assert d["pinned"] == [2] and d["label"] == REFUSED
    assert np.isfinite(d["score"]) and len(d["features"]) == 16  # still computed and reported


def test_one_sample_short_of_the_window_decides():
    d = _both(_model(), _recording(_at_rail(2, D - L + 2, D)))
    assert d["pinned"] == [] and d["label"] != REFUSED


def test_a_break_inside_the_window_resets_the_run():
    def edit(signal):
        signal[: D + 1, 3] = -RAIL  # the rail counts both signs
        signal[D - 1, 3] = RAIL * (1 - 1e-12)  # just inside the rail: the run starts again
    d = _both(_model(), _recording(edit))
    assert d["pinned"] == []


def test_exactly_rail_abs_counts_as_pinned():
    d = _both(_model(), _recording(_at_rail(5, 0, D, value=-RAIL)))
    assert d["pinned"] == [5]


def test_an_unused_channel_never_refuses():
    model = _model(unused=[6])
    assert model.used_channels == (0, 1, 2, 3, 4, 5, 7)
    d = _both(model, _recording(_at_rail(6, 0, N - 1)))
    assert d["pinned"] == [] and d["label"] != REFUSED


def test_several_pinned_channels_are_listed_in_channel_order():
    def edit(signal):
        signal[:, 7] = RAIL
        signal[:, 0] = RAIL
        signal[:, 4] = -2 * RAIL
    d = _both(_model(), _recording(edit))
    assert d["pinned"] == [0, 4, 7] and d["label"] == REFUSED


def test_the_rail_is_read_before_the_offset_is_removed():
    """The first sample is already at the rail, so the offset-removed signal is all zeros there."""
    d = _both(_model(), _recording(_at_rail(1, 0, N - 1)))
    assert d["pinned"] == [1]


def test_a_rejected_sample_does_not_touch_the_run_counters():
    """L - 1 accepted samples at the rail, plus two rejected rows at the rail inside the run
    (one with NaN on another channel, one with a bad stim): counting either would make L."""
    model = _model()
    rec = _recording(_at_rail(2, D - L + 2, D))
    dec = StreamingDecoder(model)
    out = []
    for i, (s, c) in enumerate(zip(rec.signal, rec.stim)):
        if i == D - 10:  # mid-run
            nan_row = s.copy()
            nan_row[0] = np.nan
            with pytest.raises(ValueError, match="NaN"):
                dec.push(nan_row, 0)
            with pytest.raises(ValueError, match="stim"):
                dec.push(s, 99)
        if (d := dec.push(s, c)) is not None:
            out.append(d)
    assert [(d["pinned"], d["label"] != REFUSED) for d in out] == [([], True)]


def test_rail_abs_round_trips_and_is_optional():
    model = _model(rail=700000.0)
    text = model.to_json()
    assert json.loads(text)["rail_abs"] == 700000.0
    assert Model.from_json(text).rail_abs == 700000.0
    assert "rail_abs" not in BASE.to_json() and Model.from_json(BASE.to_json()).rail_abs is None
    as_int = text.replace('"rail_abs": 700000.0', '"rail_abs": 700000')
    assert Model.from_json(as_int).rail_abs == 700000.0  # any JSON number


@pytest.mark.parametrize("value, message", [
    (0.0, "rail_abs must be finite and > 0"),
    (-1.0, "rail_abs must be finite and > 0"),
    ("1e6", "rail_abs must be a number"),
    (None, "rail_abs must be a number"),
    (True, "rail_abs must be a number"),
])
def test_bad_rail_abs_is_a_load_error(value, message):
    d = json.loads(BASE.to_json())
    d["rail_abs"] = value
    with pytest.raises(ValueError, match=message):
        Model.from_json(json.dumps(d))


def test_rail_abs_overflow_is_a_load_error():
    text = BASE.to_json().replace('"labels": ', '"rail_abs": 1e400, "labels": ', 1)
    with pytest.raises(ValueError, match="rail_abs must be finite"):
        Model.from_json(text)


def test_run_counters_saturate_at_the_window_length():
    """More than 2L samples at the rail: each counter stops at L, then a dip resets it."""
    dec = StreamingDecoder(_model())
    pinned = np.full(8, 5e5)
    pinned[2] = RAIL
    for _ in range(2 * L + 7):
        dec.push(pinned, 0)
    assert dec._run == [0, 0, L, 0, 0, 0, 0, 0]
    dec.push(np.full(8, 5e5), 0)
    assert dec._run == [0] * 8


@pytest.mark.parametrize("value", [True, False, "1e6", "700000"])
def test_model_rejects_bool_and_string_rail_abs_like_the_loader(value):
    with pytest.raises(ValueError, match="rail_abs must be a number"):
        dataclasses.replace(BASE, rail_abs=value)
