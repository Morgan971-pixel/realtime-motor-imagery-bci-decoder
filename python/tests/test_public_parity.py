"""Every core agrees with the Python reference on real EEG: EEGMMIDB subject 1 (public data,
fetched by `python -m mi_ref.eegmmidb`), at 160 Hz instead of the goldens' 250 Hz.

A model is trained on runs 4 and 8 and decodes run 12. Skipped when the data is not there
(it is never committed). Set REQUIRE_PUBLIC_DATA=1 (as CI does) to make missing data an
error. MI_PUBLIC_DATA overrides the folder (default: public-data/eegmmidb in the repo).
"""
import importlib
import os
from pathlib import Path

import numpy as np
import pytest

from mi_ref.epochs import to_samples
from mi_ref.eegmmidb import FS
from mi_ref.events import imagery_markers, markers_from_stim
from mi_ref.model import decode_recording, train
from mi_ref.recording import load_recording
from mi_ref.streaming import StreamingDecoder

CORES = ("mi_core", "mi_core_cpp")
REQUIRED = {name.strip() for name in os.environ.get("REQUIRE_CORES", "").split(",") if name.strip()}
DATA = Path(os.environ.get("MI_PUBLIC_DATA", Path(__file__).resolve().parents[2] / "public-data" / "eegmmidb"))
RUNS = [DATA / "subject001" / f"session{r:03d}" / f"S001R{r:02d}.csv" for r in (4, 8, 12)]


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    missing = [p.name for p in RUNS if not p.exists()]
    if missing:
        if os.environ.get("REQUIRE_PUBLIC_DATA") == "1":
            pytest.fail(f"REQUIRE_PUBLIC_DATA=1 but {', '.join(missing)} not found in {DATA}")
        pytest.skip(f"public data not fetched ({DATA})")
    train_recs = [load_recording(p) for p in RUNS[:2]]
    model = train(train_recs, fs=FS)
    path = tmp_path_factory.mktemp("public") / "s001_model.json"
    path.write_text(model.to_json())
    test_rec = load_recording(RUNS[2])
    return model, path, test_rec, decode_recording(model, test_rec)


@pytest.fixture(params=CORES)
def core(request):
    try:
        return importlib.import_module(request.param)
    except ImportError:
        if request.param in REQUIRED:
            raise
        pytest.skip(f"{request.param} is not built")


def _assert_same(got, expected):
    keys = ("sample", "trial", "true_label", "label", "pinned")
    assert [tuple(d[k] for k in keys) for d in got] == [tuple(e[k] for k in keys) for e in expected]
    for d, e in zip(got, expected):
        np.testing.assert_allclose(d["features"], e["features"], rtol=1e-9)
        assert abs(d["score"] - e["score"]) < 1e-7


def test_reference_decides_every_cue_of_the_held_out_run(case):
    model, _, rec, expected = case
    onsets = [m.sample for m in imagery_markers(markers_from_stim(rec.stim))]
    # each decision lands on the last sample of the 0.5 to 2.5 s window, counted at 160 Hz
    assert [d["sample"] for d in expected] == [o + to_samples(0.5, 160) + to_samples(2.0, 160) - 1 for o in onsets]
    assert len(expected) == 15  # every cue in run 12 has a full 0.5 to 2.5 s window
    assert {d["true_label"] for d in expected} == {1, 2}
    assert all(d["pinned"] == [] for d in expected)  # no rail_abs in this model


def test_python_streaming_matches_batch_on_real_eeg(case):
    model, _, rec, expected = case
    dec = StreamingDecoder(model)
    got = [d for s, c in zip(rec.signal, rec.stim) if (d := dec.push(s, c)) is not None]
    _assert_same(got, expected)


def test_core_decode_file_matches_reference_on_real_eeg(core, case):
    _, path, _, expected = case
    _assert_same(core.decode_file(str(path), str(RUNS[2])), expected)


def test_core_streaming_matches_reference_on_real_eeg(core, case):
    _, path, rec, expected = case
    dec = core.Decoder(path.read_text())
    got = [d for s, c in zip(rec.signal, rec.stim) if (d := dec.push(s, float(c))) is not None]
    _assert_same(got, expected)
