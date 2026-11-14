import json

import numpy as np
import pytest

from mi_ref.csp import VAR_FLOOR
from mi_ref.filters import filter_causal, remove_offset
from mi_ref.golden import CASES, FILTER_CHECK_SAMPLES, GOLDEN_DIR, TAIL_SAMPLES, export_golden
from mi_ref.model import REFUSED, Model, decode_recording
from mi_ref.recording import load_recording


def _check_against(golden_dir):
    """Contract tolerances: filtered abs 1e-9, features rel 1e-9, score abs 1e-7, rest exact."""
    expected = json.loads((golden_dir / "expected.json").read_text())
    assert set(expected) == set(CASES)
    for name, model_file in CASES.items():
        model = Model.from_json((golden_dir / model_file).read_text())
        stream = load_recording(golden_dir / f"{name}.csv")
        exp = expected[name]
        x = remove_offset(stream.signal)
        np.testing.assert_allclose(filter_causal(model.sos[0], x)[:FILTER_CHECK_SAMPLES], exp["band0_filtered_head"], rtol=0, atol=1e-9)
        for sos, tail in zip(model.sos, exp["filtered_tail_per_band"]):
            np.testing.assert_allclose(filter_causal(sos, x)[-TAIL_SAMPLES:], tail, rtol=0, atol=1e-9)
        got = decode_recording(model, stream)
        assert [(d["sample"], d["trial"], d["true_label"], d["label"], d["pinned"]) for d in got] == [
            (e["sample"], e["trial"], e["true_label"], e["label"], e["pinned"]) for e in exp["decisions"]
        ]
        for d, e in zip(got, exp["decisions"]):
            np.testing.assert_allclose(d["features"], e["features"], rtol=1e-9)
            assert abs(d["score"] - e["score"]) < 1e-7
    return expected


def test_committed_goldens_satisfy_the_contract():
    """The files the native cores test against must agree with this code."""
    expected = _check_against(GOLDEN_DIR)
    assert len(expected["stream"]["decisions"]) == 8
    assert np.all(np.array(expected["stream"]["band0_filtered_head"][0]) == 0.0)  # offset removed, zero state


def test_edge_stream_covers_port_traps():
    edge = json.loads((GOLDEN_DIR / "expected.json").read_text())["edge_stream"]["decisions"]
    # repeated pulse counted once, phases 1/4 ignored, truncated window skipped,
    # adjacent different codes and A,B,A each give separate markers with overlapping windows
    # and one decision lands exactly on the last sample (an onset one sample later gives none)
    assert [d["sample"] for d in edge] == [874, 1874, 2624, 2625, 2724, 2726, 4624, 7374, 9249]
    assert edge[0]["features"] == [float(np.log(VAR_FLOOR))] * 16  # flat window hits the floor


def test_rail_stream_covers_the_pinned_cases():
    model = Model.from_json((GOLDEN_DIR / "rail_model.json").read_text())
    # Oz all +0.0 and Pz all -0.0 (both unused); PO8 used only in the last band; Fz used in one
    # row of one band
    assert model.rail_abs == 1e6 and model.used_channels == (0, 1, 2, 3, 5, 7)
    assert [bool(np.any(W[:, 7] != 0.0)) for W in model.csp] == [False, False, False, True]
    assert sum(int(np.count_nonzero(W[:, 0])) for W in model.csp) == 1
    assert all(np.all(W[:, 4] == 0.0) and np.all(np.signbit(W[:, 4])) for W in model.csp)
    assert all(np.all(W[:, 6] == 0.0) and not np.any(np.signbit(W[:, 6])) for W in model.csp)
    rail = json.loads((GOLDEN_DIR / "expected.json").read_text())["rail_stream"]["decisions"]
    # 1874: C3 at the rail from sample 0 (offset-removed it reads 0); 4624: Cz for exactly the
    # window; 7374: Cz for the whole window but one sample in the middle; 10124: C4 one sample
    # short; 12874: PO8 (last band only); 15624: C3 and PO7 past the window; 18374: Pz (-0.0,
    # unused); 21124: Fz (one row); Oz at the rail throughout but unused
    assert [(d["sample"], d["pinned"]) for d in rail] == [
        (1874, [1]), (4624, [2]), (7374, []), (10124, []), (12874, [7]), (15624, [1, 5]), (18374, []), (21124, [0]),
    ]
    assert [(d["sample"], d["label"], d["pinned"]) for d in rail if d["label"] == REFUSED] == [
        (1874, 0, [1]), (4624, 0, [2]), (12874, 0, [7]), (15624, 0, [1, 5]), (21124, 0, [0]),
    ]
    assert [d["label"] == REFUSED for d in rail] == [bool(d["pinned"]) for d in rail]
    plain = json.loads((GOLDEN_DIR / "expected.json").read_text())
    assert all(d["pinned"] == [] for name in ("stream", "edge_stream") for d in plain[name]["decisions"])


def test_fresh_export_satisfies_the_same_contract(tmp_path):
    export_golden(tmp_path)
    _check_against(tmp_path)


def test_export_is_deterministic(tmp_path):
    export_golden(tmp_path / "a")
    export_golden(tmp_path / "b")
    for name in ("model.json", "stream.csv", "edge_stream.csv", "rail_model.json", "rail_stream.csv", "expected.json"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()
