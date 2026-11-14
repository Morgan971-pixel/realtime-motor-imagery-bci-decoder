import json

import numpy as np
import pytest

from mi_ref.model import Model, decode_recording, train
from mi_ref.synth import make_recording


@pytest.fixture(scope="module")
def model():
    return train([make_recording(n_trials=40, seed=0)])


def test_decoder_learns_synthetic_erd(model):
    decisions = decode_recording(model, make_recording(n_trials=20, seed=1))
    acc = np.mean([d["label"] == d["true_label"] for d in decisions])
    assert len(decisions) == 20
    assert acc >= 0.9


def test_json_roundtrip_gives_identical_decisions(model):
    rec = make_recording(n_trials=4, seed=2)
    again = Model.from_json(model.to_json())
    assert decode_recording(again, rec) == decode_recording(model, rec)


def test_json_rejects_unknown_schema(model):
    d = json.loads(model.to_json())
    d["schema_version"] = 99
    with pytest.raises(ValueError, match="schema_version"):
        Model.from_json(json.dumps(d))


def test_model_sizes(model):
    assert len(model.sos) == len(model.csp) == 4
    assert model.csp[0].shape == (4, 8)
    assert model.lda_w.shape == (16,)


def _set(path, value):
    def mutate(d):
        *parents, last = path
        for key in parents:
            d = d[key]
        d[last] = value
    return mutate


@pytest.mark.parametrize(
    "mutate, message",
    [
        (_set(["lda", "w"], [0.0, 1.0, 2.0]), "lda.w"),
        (_set(["lda", "b"], "x"), "lda.b must be a number"),
        (_set(["lda", "b"], "0.5"), "lda.b must be a number"),
        (_set(["lda", "b"], True), "lda.b must be a number"),
        (_set(["fs"], "250"), "fs must be a number"),
        (_set(["bands", 0, "low_hz"], "8"), "low_hz must be a number"),
        (_set(["bands", 0, "csp", 0, 0], "1.0"), "csp must be a number"),
        (_set(["schema_version"], True), "schema_version"),
        (_set(["schema_version"], 1.0), "schema_version"),
        (_set(["labels"], {"score_le_0": True, "score_gt_0": 2}), "labels"),
        (_set(["window", "len_s"], 0.001), "at least 2 samples"),
        (_set(["bands", 0, "csp"], [[0.0] * 5] * 4), "csp"),
        (_set(["bands", 0, "sos"], [[1.0, 2.0, 3.0]]), "sos"),
        (_set(["offset"], "mean"), "offset"),
        (_set(["channels"], ["Fz"] * 8), "channels"),
        (_set(["channels"], list(reversed(["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]))), "channels"),
        (_set(["fs"], -1.0), "fs"),
        (_set(["window", "start_s"], -1.0), "window"),
        (_set(["bands"], []), "at least one band"),
        (_set(["labels"], {"score_le_0": 2, "score_gt_0": 1}), "labels"),
        (_set(["extra"], 1), "expected keys"),
        (lambda d: d.pop("lda"), "expected keys"),
    ],
)
def test_malformed_model_is_rejected(model, mutate, message):
    d = json.loads(model.to_json())
    mutate(d)
    with pytest.raises(ValueError, match=message):
        Model.from_json(json.dumps(d))


def test_nan_is_neither_written_nor_read(model):
    with pytest.raises(ValueError):
        Model(model.fs, model.channels, model.window_start_s, model.window_len_s, model.bands,
              model.sos, model.csp, model.lda_w, float("nan"))
    text = model.to_json().replace(f'"b": {json.dumps(model.lda_b)}', '"b": NaN')
    with pytest.raises(ValueError, match="only finite numbers"):
        Model.from_json(text)


def test_repeated_keys_are_rejected(model):
    text = model.to_json().replace('"fs": ', '"fs": 250.0, "fs": ', 1)
    with pytest.raises(ValueError, match="repeats a key"):
        Model.from_json(text)
