import numpy as np
import pytest

from mi_ref.recording import COLUMNS, Recording, load_recording, save_recording
from mi_ref.synth import make_recording

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
