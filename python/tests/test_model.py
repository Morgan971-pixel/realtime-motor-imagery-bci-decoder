import json

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from helpers import native_or_none, outcome
from mi_ref.cores import NATIVE
from mi_ref.model import Model, decode_recording, train
from mi_ref.synth import make_recording
from strategies import models, streams


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


@given(model=st.booleans().flatmap(lambda realistic: models(realistic=realistic)))
def test_json_text_roundtrip_is_exact_on_generated_models(model):
    """Any valid model, extreme values included (weights up to 1e100, rail_abs up to 1e300, raw
    filter sections): writing, reading and writing again gives the same JSON text."""
    text = model.to_json()
    assert Model.from_json(text).to_json() == text


# Ways to write the same double differently: every one parses back to exactly the same value
# (17 significant digits are always enough), so a correct loader cannot tell them apart.
SPELLINGS = (
    lambda x: format(x, ".17e"),  # 5.00000000000000000e-01
    lambda x: format(x, ".17g"),  # 0.50000000000000000
    lambda x: format(x, ".17E"),  # upper-case exponent
    lambda x: repr(x),  # shortest round trip, as json.dumps writes it
)


def _respelled(value, choose):
    """JSON text for value, every float written by a spelling choose() picks (ints and the rest
    are left as JSON writes them, so schema_version and labels stay integers)."""
    if isinstance(value, float):
        return choose()(value)
    if isinstance(value, dict):
        return "{" + ", ".join(f"{json.dumps(k)}: {_respelled(v, choose)}" for k, v in value.items()) + "}"
    if isinstance(value, list):
        return "[" + ", ".join(_respelled(v, choose) for v in value) + "]"
    return json.dumps(value)


@pytest.mark.parametrize("core", tuple(NATIVE))
@given(data=st.data())
def test_native_core_reads_any_spelling_of_the_same_numbers(core, data):
    """The model JSON with every number written differently (exponent forms, 17 significant
    digits, upper-case E) loads as the same model in Python and makes a native core decide the
    same, bit for bit. A loader that does not round every number correctly (the serde_json
    float_roundtrip bug of Phase 2) fails this."""
    module = native_or_none(core)
    if module is None:
        pytest.skip(f"{NATIVE[core]} is not built")
    model = data.draw(models(realistic=False), label="model")
    signal, stim = data.draw(streams(model, realistic=False), label="stream")
    text = model.to_json()
    respelled = _respelled(json.loads(text), lambda: data.draw(st.sampled_from(SPELLINGS)))
    assert Model.from_json(respelled).to_json() == text
    assert outcome(module.Decoder(respelled), signal, stim) == outcome(module.Decoder(text), signal, stim)
