import hashlib
import io
import zipfile
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from mi_ref import cho2017
from mi_ref.cho2017 import BIOSEMI_64, FS, ONSET, PIECE, ROWS, fetch, load_eeg, to_recording
from mi_ref.eegmmidb import parse_subjects, pinned_sha256
from mi_ref.evaluate import find_subjects
from mi_ref.events import imagery_markers, markers_from_stim
from mi_ref.recording import EEG_CHANNELS, load_recording

# electrode positions (x right, y front, z up) for the 8 club rows; check_positions reads only these
POSITIONS = {"Fz": (0.0, 0.34, 0.94), "C3": (-0.49, -0.14, 0.86), "Cz": (0.0, -0.19, 0.98),
             "C4": (0.49, -0.17, 0.86), "Pz": (0.0, -0.63, 0.78), "PO7": (-0.58, -0.76, 0.30),
             "Oz": (0.0, -0.95, 0.32), "PO8": (0.45, -0.82, 0.36)}
LEFT_ONSETS, RIGHT_ONSETS = [2000, 20000], [9000, 30000]  # session order: L, R, L, R


def _hand(tags, rng):
    """68 rows x len(tags) pieces. EEG row r of piece k is common(t) + (r + 1) * tags[k], so the
    average reference leaves (r + 1 - 32.5) * tags[k]: the tag says which piece it was."""
    n = len(tags)
    common = 1e5 + 50 * rng.standard_normal(n * PIECE)
    x = np.empty((68, n * PIECE), dtype=np.float32)
    for r in range(68):
        x[r] = common + (r + 1) * np.repeat(tags, PIECE)
    return x


def _eeg_fields(**override):
    rng = np.random.default_rng(0)
    psenloc = np.zeros((64, 3))
    for ch, xyz in POSITIONS.items():
        psenloc[ROWS[ch]] = xyz
    event = np.zeros(2 * PIECE, dtype=np.uint8)
    event[ONSET::PIECE] = 1
    eeg = {"srate": 512, "n_imagery_trials": 2, "imagery_left": _hand([1.0, 2.0], rng),
           "imagery_right": _hand([101.0, 102.0], rng), "imagery_event": event, "psenloc": psenloc}
    eeg.update(override)
    return eeg


def _eeg(tmp_path, **override):
    path = tmp_path / "s99.mat"
    savemat(path, {"eeg": _eeg_fields(**override)})
    return load_eeg(path)


def test_rows_are_the_club_channels_in_the_biosemi_64_order():
    assert len(BIOSEMI_64) == len(set(BIOSEMI_64)) == 64
    assert ROWS == {"Fz": 37, "C3": 12, "Cz": 47, "C4": 49, "Pz": 30, "PO7": 24, "Oz": 28, "PO8": 61}


def test_to_recording_rebuilds_the_session_in_order_with_club_codes(tmp_path):
    rec = to_recording(_eeg(tmp_path), LEFT_ONSETS, RIGHT_ONSETS)
    assert rec.signal.shape == (4 * PIECE, 8)
    markers = imagery_markers(markers_from_stim(rec.stim))
    assert [(m.sample, m.trial, m.movement) for m in markers] == [
        (ONSET, 1, 1), (PIECE + ONSET, 2, 2), (2 * PIECE + ONSET, 3, 1), (3 * PIECE + ONSET, 4, 2)]
    for k, tag in enumerate([1.0, 101.0, 2.0, 102.0]):  # pieces in session order
        piece = rec.signal[k * PIECE:(k + 1) * PIECE]
        expected = [(ROWS[ch] + 1 - 32.5) * tag for ch in EEG_CHANNELS]  # common average removed
        np.testing.assert_allclose(piece, np.tile(expected, (PIECE, 1)), atol=0.05 * abs(tag) + 0.05)


def test_timestamps_come_from_the_true_onsets(tmp_path):
    rec = to_recording(_eeg(tmp_path), LEFT_ONSETS, RIGHT_ONSETS)
    starts = [(s - ONSET) / FS for s in sorted(LEFT_ONSETS + RIGHT_ONSETS)]
    np.testing.assert_allclose(rec.timestamps[::PIECE], starts)
    assert np.all(np.diff(rec.timestamps) > 0)
    assert rec.timestamps[ONSET] == 2000 / FS


def _swapped(a, b):
    """Positions with the rows of channels a and b exchanged: what a file in another order looks like."""
    psenloc = np.zeros((64, 3))
    for ch, xyz in POSITIONS.items():
        psenloc[ROWS[ch]] = xyz
    psenloc[[ROWS[a], ROWS[b]]] = psenloc[[ROWS[b], ROWS[a]]]
    return {"psenloc": psenloc}


@pytest.mark.parametrize("override,left,right,message", [
    ({"srate": 500}, LEFT_ONSETS, RIGHT_ONSETS, "sample rate 500 Hz, expected 512"),
    ({"imagery_event": np.roll(np.tile(np.eye(1, PIECE, ONSET, dtype=np.uint8)[0], 2), 1)}, LEFT_ONSETS,
     RIGHT_ONSETS, "imagery_event: expected one onset"),
    ({}, LEFT_ONSETS[:1], RIGHT_ONSETS, r"trial sequence \(left\): expected 2 whole-number onsets"),
    ({}, [20000, 2000], RIGHT_ONSETS, r"trial sequence \(left\): onsets must be increasing"),
    ({}, LEFT_ONSETS, [4000, 30000], "trials 1 and 2 overlap: onsets 2000 and 4000"),
    ({"psenloc": np.zeros((64, 3))}, LEFT_ONSETS, RIGHT_ONSETS, "C3 is not on the left"),
    ("swap", LEFT_ONSETS, RIGHT_ONSETS, "C3 is not on the left; C4 is not on the right"),
    ("swap_back", LEFT_ONSETS, RIGHT_ONSETS, "C3 is not in front of PO7"),
    ({}, [2000.5, 20000.0], RIGHT_ONSETS, r"trial sequence \(left\): expected 2 whole-number onsets"),
    ({"imagery_right": np.zeros((68, PIECE), dtype=np.float32)}, LEFT_ONSETS, RIGHT_ONSETS,
     "imagery_right: expected >= 64 rows x 7168 samples"),
])
def test_to_recording_refuses(tmp_path, override, left, right, message):
    if override == "swap":
        override = _swapped("C3", "C4")
    elif override == "swap_back":  # both on the left, so only the front-to-back rule sees it
        override = _swapped("C3", "PO7")
    with pytest.raises(ValueError, match=message):
        to_recording(_eeg(tmp_path, **override), left, right)


def test_a_shifted_or_tilted_cap_is_accepted(tmp_path):
    """Digitised caps sit off-centre (one person's midline is at x = -0.2) or tilted back (Fz
    higher than Cz); the order of the channels is what matters."""
    psenloc = np.zeros((64, 3))
    for ch, (x, y, z) in POSITIONS.items():
        psenloc[ROWS[ch]] = (x - 0.2, y + 0.15, z - 0.1 if ch == "Cz" else z)
    shifted = to_recording(_eeg(tmp_path, psenloc=psenloc), LEFT_ONSETS, RIGHT_ONSETS)
    centred = to_recording(_eeg(tmp_path), LEFT_ONSETS, RIGHT_ONSETS)
    np.testing.assert_array_equal(shifted.signal, centred.signal)  # positions are checked, never used
    np.testing.assert_array_equal(shifted.stim, centred.stim)


def test_whole_number_onsets_stored_as_floats_are_accepted(tmp_path):
    as_floats = to_recording(_eeg(tmp_path), np.array(LEFT_ONSETS, float), np.array(RIGHT_ONSETS, float))
    np.testing.assert_array_equal(as_floats.stim, to_recording(_eeg(tmp_path), LEFT_ONSETS, RIGHT_ONSETS).stim)


def test_to_recording_refuses_a_nan_in_the_eeg(tmp_path):
    left = _hand([1.0, 2.0], np.random.default_rng(1))
    left[5, 100] = np.nan
    with pytest.raises(ValueError, match="imagery_left: contains a NaN"):
        to_recording(_eeg(tmp_path, imagery_left=left), LEFT_ONSETS, RIGHT_ONSETS)


def _mat_bytes(fields) -> bytes:
    buf = io.BytesIO()
    savemat(buf, fields)
    return buf.getvalue()


class FakeServer:
    """Serves synthetic files by URL and records each request."""

    def __init__(self, files: dict[str, bytes]):
        self.files, self.requests = files, []

    def get(self, url: str, dest: Path) -> None:
        self.requests.append(url)
        dest.write_bytes(self.files[url.removeprefix(cho2017.BASE_URL + "/")])


def _server(subjects, bad=None) -> tuple[FakeServer, dict[str, str]]:
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w") as zf:
        for s in subjects:
            seq = {"imagery_left": np.array(LEFT_ONSETS), "imagery_right": np.array(RIGHT_ONSETS)}
            zf.writestr(f"s{s}_trial_sequence_v1.mat", _mat_bytes({"trial_sequence": seq}))
    files = {"trial_sequence.zip": zbuf.getvalue()}
    for s in subjects:
        files[f"mat_data/s{s:02d}.mat"] = _mat_bytes({"eeg": _eeg_fields(**({"srate": 500} if s == bad else {}))})
    return FakeServer(files), {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}


def test_fetch_converts_each_person_into_the_club_layout(tmp_path):
    server, sums = _server([1, 12])
    assert fetch(tmp_path, [1, 12], get=server.get, sums=sums) == {"converted": 2, "present": 0, "refused": 0}
    assert server.requests == [f"{cho2017.BASE_URL}/trial_sequence.zip", f"{cho2017.BASE_URL}/mat_data/s01.mat",
                               f"{cho2017.BASE_URL}/mat_data/s12.mat"]
    rec = load_recording(tmp_path / "subject012" / "session001" / "s12.csv")
    expected = to_recording(_eeg(tmp_path), LEFT_ONSETS, RIGHT_ONSETS)
    np.testing.assert_array_equal(rec.stim, expected.stim)
    np.testing.assert_allclose(rec.signal, expected.signal, atol=1e-6)  # the CSV keeps 6 decimals
    assert sorted(p.name for p in (tmp_path / "raw").iterdir()) == ["trial_sequence.zip"]  # .mat deleted
    subjects, skipped = find_subjects(tmp_path, min_trials=4)
    assert sorted(subjects) == ["subject001", "subject012"] and skipped == []


def test_fetch_is_resumable_and_keeps_mat_on_request(tmp_path):
    server, sums = _server([1])
    fetch(tmp_path, [1], get=server.get, sums=sums, keep_mat=True)
    assert sorted(p.name for p in (tmp_path / "raw").iterdir()) == ["s01.mat", "trial_sequence.zip"]
    assert fetch(tmp_path, [1], get=server.get, sums=sums) == {"converted": 0, "present": 1, "refused": 0}
    (tmp_path / "subject001" / "session001" / "s01.csv").unlink()
    assert fetch(tmp_path, [1], get=server.get, sums=sums)["converted"] == 1
    assert len(server.requests) == 2  # the kept .mat and zip were reused, not downloaded again


def test_fetch_records_a_refused_person_and_does_not_retry_them(tmp_path, capsys):
    server, sums = _server([1, 2], bad=2)
    assert fetch(tmp_path, [1, 2], get=server.get, sums=sums) == {"converted": 1, "present": 0, "refused": 1}
    assert "sample rate 500 Hz" in (tmp_path / "subject002" / "session001" / "s02.refused").read_text()
    assert (tmp_path / "raw" / "s02.mat").exists() and not (tmp_path / "raw" / "s01.mat").exists()
    assert "s02: refused: sample rate 500 Hz" in capsys.readouterr().err
    assert fetch(tmp_path, [1, 2], get=server.get, sums=sums)["present"] == 2


def test_fetch_stops_on_a_checksum_mismatch_and_deletes_the_file(tmp_path):
    server, sums = _server([1, 2])
    sums["mat_data/s02.mat"] = "0" * 64
    with pytest.raises(ValueError, match="mat_data/s02.mat: SHA-256 does not match"):
        fetch(tmp_path, [1, 2], get=server.get, sums=sums)
    assert not (tmp_path / "raw" / "s02.mat").exists()
    assert (tmp_path / "subject001" / "session001" / "s01.csv").exists()


def test_fetch_refuses_a_file_without_a_pinned_checksum(tmp_path):
    server, sums = _server([1])
    del sums["mat_data/s01.mat"]
    with pytest.raises(ValueError, match="mat_data/s01.mat: no pinned SHA-256"):
        fetch(tmp_path, [1], get=server.get, sums=sums)


def test_a_person_missing_from_the_trial_sequences_is_refused(tmp_path):
    server, sums = _server([1])
    server.files["mat_data/s03.mat"] = server.files["mat_data/s01.mat"]
    sums["mat_data/s03.mat"] = sums["mat_data/s01.mat"]
    assert fetch(tmp_path, [3], get=server.get, sums=sums)["refused"] == 1
    assert "has no s3_trial_sequence_v1.mat" in (tmp_path / "subject003" / "session001" / "s03.refused").read_text()


@pytest.mark.parametrize("content,kind", [
    (_mat_bytes({"eeg": {"srate": 512}}), "AttributeError"),  # a struct without the dataset's fields
    (b"not a MATLAB file at all", "IndexError"),
])
def test_an_unreadable_file_is_refused_and_the_fetch_goes_on(tmp_path, content, kind):
    server, sums = _server([1, 2])
    server.files["mat_data/s01.mat"] = content
    sums["mat_data/s01.mat"] = hashlib.sha256(content).hexdigest()
    assert fetch(tmp_path, [1, 2], get=server.get, sums=sums) == {"converted": 1, "present": 0, "refused": 1}
    reason = (tmp_path / "subject001" / "session001" / "s01.refused").read_text()
    assert reason.startswith(f"s01.mat: not readable as this dataset ({kind}")


def test_a_trial_sequence_checksum_mismatch_stops_the_fetch(tmp_path):
    server, sums = _server([1])
    sums["trial_sequence.zip"] = "0" * 64
    with pytest.raises(ValueError, match="trial_sequence.zip: SHA-256 does not match"):
        fetch(tmp_path, [1], get=server.get, sums=sums)
    assert not (tmp_path / "raw" / "trial_sequence.zip").exists()


def test_main_reports_a_failed_fetch_and_exits_1(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cho2017, "download", lambda url, dest: dest.write_bytes(b"tampered"))
    assert cho2017.main(["--out", str(tmp_path), "--subjects", "1"]) == 1
    assert "error: trial_sequence.zip: SHA-256 does not match the pinned value" in capsys.readouterr().err


def test_subjects_are_1_to_52():
    assert parse_subjects("1-52", cho2017.SUBJECTS) == list(range(1, 53))
    with pytest.raises(ValueError, match="within 1-52"):
        parse_subjects("50-53", cho2017.SUBJECTS)
    with pytest.raises(SystemExit):
        cho2017.main(["--out", "unused", "--subjects", "0"])


def test_pinned_sha256_covers_every_file():
    sums = pinned_sha256(cho2017.SHA256_FILE)
    assert set(sums) == {"trial_sequence.zip"} | {f"mat_data/s{s:02d}.mat" for s in range(1, 53)}
    assert sums["mat_data/s01.mat"] == "57f2f10056b3c240adc78324872597d9b06b282df537a7763e98467275efe6db"
