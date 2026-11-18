"""Every native core agrees with the Python reference on Cho 2017 person 1 (public data, fetched by
`python -m mi_ref.cho2017 --out ../public-data/cho2017 --subjects 1`), at 512 Hz: the first rate
above the club's 250 Hz, with a 1024-sample window.

A model is trained on the first 150 trials and decodes the last 50, written to their own CSV.
Skipped when the data is not there (it is never committed, and CI does not fetch it); set
REQUIRE_CHO2017_DATA=1 to make missing data an error. MI_CHO2017_DATA overrides the folder
(default: public-data/cho2017 in the repo).
"""
import os
from pathlib import Path

import pytest

from helpers import assert_same_decisions
from mi_ref.cho2017 import FS, ONSET, PIECE
from mi_ref.epochs import to_samples
from mi_ref.model import decode_recording, train
from mi_ref.recording import Recording, load_recording, save_recording

DATA = Path(os.environ.get("MI_CHO2017_DATA", Path(__file__).resolve().parents[2] / "public-data" / "cho2017"))
CSV = DATA / "subject001" / "session001" / "s01.csv"
TRAIN_TRIALS = 150


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    if not CSV.exists():
        if os.environ.get("REQUIRE_CHO2017_DATA") == "1":
            pytest.fail(f"REQUIRE_CHO2017_DATA=1 but {CSV} not found")
        pytest.skip(f"Cho 2017 person 1 not fetched ({CSV})")
    rec = load_recording(CSV)
    cut = TRAIN_TRIALS * PIECE  # every trial is one piece, so this is a trial boundary

    def part(sl):
        return Recording(rec.timestamps[sl], rec.signal[sl], rec.stim[sl])

    model = train([part(slice(0, cut))], fs=FS)
    tmp = tmp_path_factory.mktemp("cho2017")
    model_path, held_out_csv = tmp / "s01_model.json", tmp / "s01_held_out.csv"
    model_path.write_text(model.to_json())
    held_out = part(slice(cut, None))
    save_recording(held_out_csv, held_out)
    held_out = load_recording(held_out_csv)  # decode exactly what the cores read
    return model_path, held_out_csv, held_out, decode_recording(model, held_out)


def test_reference_decides_every_held_out_trial(case):
    _, _, rec, expected = case
    assert len(expected) == 200 - TRAIN_TRIALS
    window_end = ONSET + to_samples(0.5, FS) + to_samples(2.0, FS) - 1  # 1023 + 256 + 1024 - 1
    assert [d["sample"] for d in expected] == [k * PIECE + window_end for k in range(len(expected))]
    assert [d["trial"] for d in expected] == list(range(TRAIN_TRIALS + 1, 201))
    assert {d["true_label"] for d in expected} == {1, 2}


def test_core_decode_file_matches_reference_at_512_hz(native_core, case):
    model_path, csv, _, expected = case
    assert_same_decisions(native_core.decode_file(str(model_path), str(csv)), expected)


def test_core_streaming_matches_reference_at_512_hz(native_core, case):
    model_path, _, rec, expected = case
    dec = native_core.Decoder(model_path.read_text())
    got = [d for s, c in zip(rec.signal, rec.stim) if (d := dec.push(s, float(c))) is not None]
    assert_same_decisions(got, expected)
