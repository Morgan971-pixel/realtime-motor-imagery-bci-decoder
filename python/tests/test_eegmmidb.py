import hashlib
from pathlib import Path

import numpy as np
import pytest

from mi_ref import eegmmidb
from mi_ref.edf import read_edf, write_edf
from mi_ref.eegmmidb import IMAGERY_RUNS, fetch, parse_subjects, pinned_sha256, to_recording
from mi_ref.evaluate import find_subjects
from mi_ref.events import imagery_markers, markers_from_stim
from mi_ref.recording import EEG_CHANNELS, load_recording

# EEGMMIDB order and spelling for the 8 club channels, plus two the converter must ignore
EDF_LABELS = ("Fc5.", "C3..", "Cz..", "C4..", "Fz..", "Pz..", "Po7.", "Oz..", "Po8.", "Iz..")
CUES = [(0.0, 4.2, "T0"), (4.2, 4.1, "T2"), (8.3, 4.2, "T0"), (12.5, 4.1, "T1"), (16.6, 4.2, "T0")]


def _edf_bytes(labels=EDF_LABELS, fs=160.0, seconds=20, cues=CUES, **kw) -> bytes:
    n = int(fs * seconds)
    signals = {label: (np.arange(n) * (k + 1)) % 200 - 100 for k, label in enumerate(labels)}
    return write_edf(signals, fs, cues, **kw)


def test_to_recording_picks_club_channels_in_club_order():
    edf = read_edf(_edf_bytes())
    rec = to_recording(edf, 4)
    assert rec.signal.shape == (3200, 8)
    for c, ch in enumerate(EEG_CHANNELS):
        src = [i for i, label in enumerate(edf.labels) if label.rstrip(".").upper() == ch.upper()][0]
        np.testing.assert_array_equal(rec.signal[:, c], edf.signals[src])
    np.testing.assert_array_equal(rec.timestamps, np.arange(3200) / 160.0)


def test_to_recording_writes_club_imagery_codes_at_cue_onsets():
    rec = to_recording(read_edf(_edf_bytes()), 8)
    assert np.flatnonzero(rec.stim).tolist() == [672, 2000]  # 4.2 s and 12.5 s at 160 Hz
    assert rec.stim[672] == 1 * 100 + 30 + 2  # trial 1, imagery, right (T2)
    assert rec.stim[2000] == 2 * 100 + 30 + 1  # trial 2, imagery, left (T1)
    markers = imagery_markers(markers_from_stim(rec.stim))
    assert [(m.sample, m.trial, m.movement) for m in markers] == [(672, 1, 2), (2000, 2, 1)]


@pytest.mark.parametrize("kwargs,run,message", [
    ({}, 6, "not a left/right imagery run"),
    ({"labels": EDF_LABELS[:-2]}, 4, "channel PO8: expected exactly one signal, found 0"),
    ({"labels": ("Cz..",) + EDF_LABELS[1:2] + ("CZ",) + EDF_LABELS[3:]}, 4, "channel Cz: expected exactly one signal, found 2"),
    ({"fs": 128.0}, 4, "sample rate 128 Hz, expected 160"),
    ({"unit": "mV"}, 4, "unit 'mV'"),
    ({"cues": [(1.0, 4.1, "T3")]}, 4, "unexpected annotation 'T3'"),
    ({"cues": [(0.0, 4.2, "T0")]}, 4, "no T1/T2 cues"),
    ({"cues": [(1.0, 4.1, "T1"), (1.001, 4.1, "T2")]}, 4, "not after the previous cue"),
])
def test_to_recording_refuses(kwargs, run, message):
    with pytest.raises(ValueError, match=message):
        to_recording(read_edf(_edf_bytes(**kwargs)), run)


def test_to_recording_refuses_cue_past_the_end():
    data = _edf_bytes(seconds=2, cues=[(1.9999, 4.1, "T1")])  # rounds to sample 320 of 320
    with pytest.raises(ValueError, match="past the end"):
        to_recording(read_edf(data), 4)


def test_parse_subjects():
    assert parse_subjects("1-3,7,2") == [1, 2, 3, 7]
    assert parse_subjects("1-109") == list(range(1, 110))
    for bad in ("0", "110", "3-1", "a", "1-", "-4", "1,,2", " 1"):
        with pytest.raises(ValueError):
            parse_subjects(bad)


def test_pinned_sha256_covers_every_imagery_run():
    sums = pinned_sha256()
    expected = {f"S{s:03d}/S{s:03d}R{r:02d}.edf" for s in range(1, 110) for r in IMAGERY_RUNS}
    assert set(sums) == expected
    assert all(len(d) == 64 and int(d, 16) >= 0 for d in sums.values())


class FakeServer:
    """Serves synthetic EDF files by URL and records each request."""

    def __init__(self, files: dict[str, bytes]):
        self.files, self.requests = files, []

    def get(self, url: str, dest: Path) -> None:
        self.requests.append(url)
        dest.write_bytes(self.files[url.removeprefix(eegmmidb.BASE_URL + "/")])


def _server(subjects, bad_run=None) -> tuple[FakeServer, dict[str, str]]:
    files = {}
    for s in subjects:
        for r in IMAGERY_RUNS:
            fs = 128.0 if (s, r) == bad_run else 160.0
            files[f"S{s:03d}/S{s:03d}R{r:02d}.edf"] = _edf_bytes(fs=fs)
    return FakeServer(files), {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}


def test_fetch_converts_every_run_into_the_club_layout(tmp_path):
    server, sums = _server([1, 2])
    counts = fetch(tmp_path, [1, 2], get=server.get, sums=sums)
    assert counts == {"converted": 6, "present": 0, "refused": 0}
    assert server.requests[0] == "https://physionet.org/files/eegmmidb/1.0.0/S001/S001R04.edf"
    csv = tmp_path / "subject002" / "session012" / "S002R12.csv"
    rec = load_recording(csv)
    np.testing.assert_array_equal(rec.stim, to_recording(read_edf(server.files["S002/S002R12.edf"]), 12).stim)
    assert list((tmp_path / "edf").iterdir()) == []  # EDF deleted after conversion
    subjects, skipped = find_subjects(tmp_path, min_trials=2)
    assert sorted(subjects) == ["subject001", "subject002"] and skipped == []
    assert [len(recs) for recs in subjects.values()] == [3, 3]


def test_fetch_is_resumable_and_keeps_edf_on_request(tmp_path):
    server, sums = _server([1])
    fetch(tmp_path, [1], get=server.get, sums=sums, keep_edf=True)
    assert sorted(p.name for p in (tmp_path / "edf").iterdir()) == ["S001R04.edf", "S001R08.edf", "S001R12.edf"]
    assert fetch(tmp_path, [1], get=server.get, sums=sums) == {"converted": 0, "present": 3, "refused": 0}
    assert len(server.requests) == 3
    (tmp_path / "subject001" / "session008" / "S001R08.csv").unlink()
    assert fetch(tmp_path, [1], get=server.get, sums=sums)["converted"] == 1
    assert len(server.requests) == 3  # the kept EDF was reused, not downloaded again


def test_fetch_records_a_refused_run_and_does_not_retry_it(tmp_path, capsys):
    server, sums = _server([1], bad_run=(1, 8))
    assert fetch(tmp_path, [1], get=server.get, sums=sums) == {"converted": 2, "present": 0, "refused": 1}
    marker = tmp_path / "subject001" / "session008" / "S001R08.refused"
    assert "sample rate 128 Hz" in marker.read_text()
    assert "S001R08: refused: channel Fz: sample rate 128 Hz" in capsys.readouterr().err
    assert fetch(tmp_path, [1], get=server.get, sums=sums)["present"] == 3


def test_fetch_stops_on_a_checksum_mismatch_and_deletes_the_file(tmp_path):
    server, sums = _server([1])
    sums["S001/S001R08.edf"] = "0" * 64
    with pytest.raises(ValueError, match="S001/S001R08.edf: SHA-256 does not match"):
        fetch(tmp_path, [1], get=server.get, sums=sums)
    assert not (tmp_path / "edf" / "S001R08.edf").exists()
    assert (tmp_path / "subject001" / "session004" / "S001R04.csv").exists()


def test_fetch_refuses_a_run_without_a_pinned_checksum(tmp_path):
    server, sums = _server([1])
    del sums["S001/S001R04.edf"]
    with pytest.raises(ValueError, match="no pinned SHA-256"):
        fetch(tmp_path, [1], get=server.get, sums=sums)
    assert server.requests == []


def test_main_rejects_a_bad_subject_list(capsys):
    with pytest.raises(SystemExit) as exc:
        eegmmidb.main(["--out", "x", "--subjects", "0-3"])
    assert exc.value.code == 2
    assert "within 1-109" in capsys.readouterr().err


def test_main_reports_a_dropped_connection_without_a_traceback(tmp_path, monkeypatch, capsys):
    import http.client

    def drop(url, dest):
        raise http.client.IncompleteRead(b"partial", 100)

    monkeypatch.setattr(eegmmidb, "download", drop)
    assert eegmmidb.main(["--out", str(tmp_path), "--subjects", "1"]) == 1
    assert capsys.readouterr().err.startswith("error: ")


@pytest.mark.parametrize("line,message", [
    ("abc S001/S001R04.edf", "not a 64-digit hex digest"),
    ("0" * 64 + " S001/S001R04.edf extra", "expected '<digest> <name>'"),
    ("0" * 64, "expected '<digest> <name>'"),
])
def test_pinned_sha256_refuses_malformed_lines(tmp_path, line, message):
    path = tmp_path / "sums.txt"
    path.write_text(line + "\n")
    with pytest.raises(ValueError, match=message):
        pinned_sha256(path)
