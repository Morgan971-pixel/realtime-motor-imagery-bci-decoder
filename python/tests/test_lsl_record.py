import json
import os
import threading
import uuid
from pathlib import Path

import numpy as np
import pytest

from mi_ref import lsl_record
from mi_ref.cues import marker_info
from mi_ref.events import markers_from_stim
from mi_ref.golden import GOLDEN_DIR
from mi_ref.lsl_live import open_streams, parse_channels
from mi_ref.lsl_record import record
from mi_ref.lsl_replay import publish_split, split_stream_info
from mi_ref.lsl_source import LiveSource
from mi_ref.model import Model, decode_recording
from mi_ref.paths import REPO, outside_repo
from mi_ref.recording import load_recording, save_recording

try:
    import pylsl
except ImportError:
    if os.environ.get("REQUIRE_LSL") == "1":
        raise
    pytest.skip("pylsl is not installed", allow_module_level=True)

MODEL = Model.from_json((GOLDEN_DIR / "model.json").read_text())
GOLDEN = load_recording(GOLDEN_DIR / "stream.csv")
EXPECTED = json.loads((GOLDEN_DIR / "expected.json").read_text())["stream"]["decisions"]


def test_outside_repo_refuses_the_repo_and_anything_in_it(tmp_path):
    assert outside_repo(tmp_path / "x.csv") == (tmp_path / "x.csv").resolve()
    for path in (REPO, REPO / "x.csv", REPO / "public-data" / "s1.csv", REPO / "python" / ".." / "y.json"):
        with pytest.raises(ValueError, match="stay outside the repository"):
            outside_repo(path)
    link = tmp_path / "link"
    link.symlink_to(REPO)
    with pytest.raises(ValueError, match="stay outside the repository"):
        outside_repo(link / "z.csv")


def test_a_recorded_session_decodes_offline_exactly_like_live(tmp_path):
    tag = uuid.uuid4().hex[:12]
    positions = (8, 1, 2, 3, 4, 5, 6, 7)
    eeg = pylsl.StreamOutlet(split_stream_info(f"eeg-{tag}", f"eeg-{tag}", MODEL.fs, pylsl, positions))
    cues = pylsl.StreamOutlet(marker_info(f"cues-{tag}", f"cues-{tag}", pylsl))
    inlet, eeg_indices, stim_index, cue_inlet = open_streams(
        pylsl, MODEL.fs, f"eeg-{tag}", "EEG", parse_channels("8,1-7"), f"cues-{tag}", 5)
    publisher = threading.Thread(target=publish_split, args=(GOLDEN, MODEL), kwargs=dict(
        eeg_outlet=eeg, cue_outlet=cues, pylsl_module=pylsl, eeg_positions=positions, paced=False, cues_first=True))
    publisher.start()
    rec = record(LiveSource(inlet, fs=MODEL.fs, eeg_indices=eeg_indices, stim_index=stim_index,
                            cue_inlet=cue_inlet, max_samples=len(GOLDEN.stim), timeout=2, pylsl_module=pylsl,
                            report=pytest.fail))
    publisher.join(timeout=10)
    np.testing.assert_array_equal(rec.signal, GOLDEN.signal)
    assert markers_from_stim(rec.stim) == markers_from_stim(GOLDEN.stim)
    assert np.all(np.diff(rec.timestamps) > 0)
    path = tmp_path / "session.csv"
    save_recording(path, rec)
    decisions = decode_recording(MODEL, load_recording(path))
    keys = ("sample", "trial", "true_label", "label")
    assert [tuple(d[k] for k in keys) for d in decisions] == [tuple(e[k] for k in keys) for e in EXPECTED]


def test_main_refuses_an_output_inside_the_repo_or_an_existing_file(tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        lsl_record.main(["--out", str(REPO / "results-local" / "s.csv"), "--seconds", "1"])
    assert e.value.code == 2 and "stay outside the repository" in capsys.readouterr().err
    existing = tmp_path / "s.csv"
    existing.write_text("x")
    with pytest.raises(SystemExit) as e:
        lsl_record.main(["--out", str(existing), "--seconds", "1"])
    assert e.value.code == 2 and "already exists" in capsys.readouterr().err
    assert existing.read_text() == "x"


class Pulls:
    """A stand-in inlet that serves fixed pulls in order; an exception in the list is raised."""

    def __init__(self, pulls):
        self.pulls, self.closed = list(pulls), False

    def open_stream(self, timeout):
        pass

    def pull_chunk(self, timeout, max_samples=1024):
        pull = self.pulls.pop(0) if self.pulls else ([], [])
        if isinstance(pull, BaseException):
            raise pull
        return pull

    def close_stream(self):
        self.closed = True


def test_ctrl_c_saves_the_held_samples_with_the_cues_that_already_arrived():
    """Ctrl+C while the cue merger still holds every sample: they are saved, and a cue that
    arrived after the last pull still lands on its sample."""
    fs = MODEL.fs
    eeg = Pulls([([[float(i)] * 8 for i in range(10)], [100 + i / fs for i in range(10)]), KeyboardInterrupt()])
    cue = Pulls([([], []), ([[231]], [100 + 4 / fs])])
    messages = []
    rec = record(LiveSource(eeg, eeg_indices=list(range(8)), cue_inlet=cue, cue_lag=1.0, fs=fs, timeout=0.01,
                            pylsl_module=pylsl, report=messages.append))
    assert messages == ["stopped early; saving what was recorded"]
    np.testing.assert_array_equal(rec.signal[:, 0], np.arange(10.0))
    assert [i for i, s in enumerate(rec.stim) if s] == [4] and rec.stim[4] == 231
    assert eeg.closed and cue.closed


def test_outside_repo_is_not_fooled_by_letter_case(tmp_path):
    shouted = Path(str(REPO).upper())
    if not shouted.exists():
        pytest.skip("case-sensitive filesystem")
    with pytest.raises(ValueError, match="stay outside the repository"):
        outside_repo(shouted / "x.csv")


def test_write_new_never_overwrites(tmp_path):
    path = tmp_path / "s.csv"
    lsl_record.write_new(path, GOLDEN)
    assert load_recording(path).signal.shape == GOLDEN.signal.shape
    with pytest.raises(FileExistsError):
        lsl_record.write_new(path, GOLDEN)
    assert load_recording(path).signal.shape == GOLDEN.signal.shape
