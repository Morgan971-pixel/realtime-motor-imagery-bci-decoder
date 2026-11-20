import tempfile
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from mi_ref.recording import COLUMNS, EEG_CHANNELS, Recording, load_recording, save_recording
from mi_ref.synth import make_recording
from strategies import any_codes, models, streams

HEADER = ",".join(COLUMNS)


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def test_save_load_roundtrip(tmp_path):
    rec = make_recording(n_trials=2, seed=3)
    path = tmp_path / "rec.csv"
    save_recording(path, rec)
    back = load_recording(path)
    np.testing.assert_allclose(back.signal, rec.signal, atol=1e-6)
    np.testing.assert_allclose(back.timestamps, rec.timestamps, atol=1e-6)
    np.testing.assert_array_equal(back.stim, rec.stim)


def test_load_rejects_wrong_header(tmp_path):
    with pytest.raises(ValueError, match="unexpected columns"):
        load_recording(_write(tmp_path / "bad.csv", "time,a,b\n1,2,3\n"))


def test_load_accepts_byte_order_mark_and_spaces(tmp_path):
    row = "1.0," + ",".join(["5.0"] * 8) + ",0"
    path = _write(tmp_path / "bom.csv", "\ufeff" + HEADER.replace(",", ", ") + "\n" + row + "\n")
    assert load_recording(path).signal.shape == (1, 8)


def test_load_rejects_header_only_file(tmp_path):
    with pytest.raises(ValueError, match="no samples"):
        load_recording(_write(tmp_path / "empty.csv", HEADER + "\n"))


def test_load_rejects_short_rows(tmp_path):
    with pytest.raises(ValueError, match="expected 10 columns"):
        load_recording(_write(tmp_path / "short.csv", HEADER + "\n1.0,2.0,3.0\n"))


@pytest.mark.parametrize("stim, message", [("100.4", "non-integer"), ("nan", "NaN or infinite")])
def test_load_rejects_bad_stim_instead_of_rounding(tmp_path, stim, message):
    row = "1.0," + ",".join(["5.0"] * 8) + "," + stim
    with pytest.raises(ValueError, match=message):
        load_recording(_write(tmp_path / "stim.csv", HEADER + "\n" + row + "\n"))


def test_recording_rejects_float_stim():
    with pytest.raises(ValueError, match="stim must be integer"):
        Recording(np.zeros(2), np.zeros((2, 8)), np.zeros(2))


def test_recording_rejects_non_finite_eeg():
    signal = np.zeros((2, 8))
    signal[1, 3] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        Recording(np.zeros(2), signal, np.zeros(2, dtype=np.int64))


def test_load_rejects_comment_lines(tmp_path):
    """The CSV dialect has no comments: a '#' line is an error, as in the Rust reader."""
    row = "1.0," + ",".join(["5.0"] * 8) + ",0"
    with pytest.raises(ValueError):
        load_recording(_write(tmp_path / "c.csv", HEADER + "\n# a note\n" + row + "\n"))


# values whose 6-decimal form is a corner: negative zero, tiny negatives that print as -0.000000,
# and values that round up across a power of ten (9.9999996 -> 10.000000)
_CORNERS = [-0.0, -4e-7, 4e-7, 5e-7, -5.000001e-7, 9.9999996, -99.9999997, 999999.9999996, 0.0999999996]


def _six_decimals(x):
    """The oracle: each value as "%.6f" text, read back by Python's float()."""
    return np.array([float(f"{v:.6f}") for v in np.ravel(x)]).reshape(np.shape(x))


def _same_bits(got, expected):
    """Elementwise equal, and 0.0 differs from -0.0."""
    np.testing.assert_array_equal(np.asarray(got, dtype=np.float64).view(np.uint64),
                                  np.asarray(expected, dtype=np.float64).view(np.uint64))


@st.composite
def _signals(draw):
    """(n, 8) EEG: a decoder stream on a realistic model, or values drawn directly with corners."""
    if draw(st.booleans()):
        signal, _ = draw(streams(draw(models(realistic=True))))
        return signal
    n = draw(st.integers(1, 50))
    value = st.one_of(st.floats(-1e9, 1e9, allow_nan=False), st.sampled_from(_CORNERS))
    return np.array(draw(st.lists(value, min_size=n * len(EEG_CHANNELS), max_size=n * len(EEG_CHANNELS)))).reshape(n, -1)


@given(data=st.data())
def test_save_load_rounds_every_value_to_six_decimals(data):
    """A saved and loaded recording holds each signal value and timestamp rounded to exactly 6
    decimals (sign of zero included) and every stim code unchanged."""
    signal = data.draw(_signals(), label="signal")
    n = len(signal)
    start = data.draw(st.one_of(st.floats(0.0, 2e9), st.sampled_from(_CORNERS[:6])), label="t0")
    steps = data.draw(st.lists(st.floats(1e-3, 10.0), min_size=n - 1, max_size=n - 1), label="steps")
    timestamps = np.concatenate([[start], start + np.cumsum(steps)])  # keeps a -0.0 start
    stim = np.array(data.draw(st.lists(st.one_of(st.just(0), any_codes()), min_size=n, max_size=n), label="stim"),
                    dtype=np.int64)
    rec = Recording(timestamps, signal, stim)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "rec.csv"
        save_recording(path, rec)
        back = load_recording(path)
    _same_bits(back.signal, _six_decimals(signal))
    _same_bits(back.timestamps, _six_decimals(timestamps))
    np.testing.assert_array_equal(back.stim, stim)
    assert back.stim.dtype == np.int64
