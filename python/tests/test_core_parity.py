"""Every native core (Rust `mi_core`, C++ `mi_core_cpp`) must agree with the Python reference.

A core that is not built is skipped, so the Python suite runs on its own. Set
REQUIRE_CORES=mi_core,mi_core_cpp (for example in CI) to make a missing build an error.
"""
import importlib
import json
import os
import re

import numpy as np
import pytest

from mi_ref.golden import CASES, GOLDEN_DIR
from mi_ref.invalid import INVALID_DIR, expand_rows, to_float
from mi_ref.model import Model, decode_recording
from mi_ref.recording import Recording, load_recording, save_recording
from mi_ref.synth import make_recording

CORES = ("mi_core", "mi_core_cpp")
REQUIRED = {name.strip() for name in os.environ.get("REQUIRE_CORES", "").split(",") if name.strip()}
assert REQUIRED <= set(CORES), f"unknown core in REQUIRE_CORES: {sorted(REQUIRED - set(CORES))}"
MODEL_TEXT = (GOLDEN_DIR / "model.json").read_text()
MANIFEST = json.loads((INVALID_DIR / "manifest.json").read_text())


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


@pytest.mark.parametrize("name", CASES)
def test_core_reproduces_committed_goldens(core, name):
    expected = json.loads((GOLDEN_DIR / "expected.json").read_text())[name]["decisions"]
    _assert_same(core.decode_file(str(GOLDEN_DIR / CASES[name]), str(GOLDEN_DIR / f"{name}.csv")), expected)


def test_streaming_binding_refuses_pinned_windows(core):
    rec = load_recording(GOLDEN_DIR / "rail_stream.csv")
    dec = core.Decoder((GOLDEN_DIR / "rail_model.json").read_text())
    got = [d for s, c in zip(rec.signal, rec.stim) if (d := dec.push(s, float(c))) is not None]
    assert [(d["sample"], d["label"], d["pinned"]) for d in got if d["pinned"]] == [
        (1874, 0, [1]), (4624, 0, [2]), (12874, 0, [7]), (15624, 0, [1, 5]), (21124, 0, [0]),
    ]
    assert [d["label"] == 0 for d in got] == [bool(d["pinned"]) for d in got]
    _assert_same(got, core.decode_file(str(GOLDEN_DIR / "rail_model.json"), str(GOLDEN_DIR / "rail_stream.csv")))


@pytest.mark.parametrize("seed", range(3))
def test_core_equals_python_on_random_marker_patterns(core, tmp_path, seed):
    rng = np.random.default_rng(seed)
    base = make_recording(n_trials=2, seed=200 + seed)
    stim = np.zeros(len(base.stim), dtype=np.int64)
    for i in rng.choice(len(stim) - 3, size=30, replace=False):
        stim[i : i + int(rng.integers(1, 4))] = int(rng.choice([131, 132, 121, 151, 111, 141, 232, 331]))
    path = tmp_path / "rec.csv"
    save_recording(path, Recording(base.timestamps, base.signal, stim))
    on_disk = load_recording(path)  # every core reads the same rounded CSV values
    _assert_same(core.decode_file(str(GOLDEN_DIR / "model.json"), str(path)), decode_recording(Model.from_json(MODEL_TEXT), on_disk))


def test_streaming_binding_matches_batch(core):
    rec = load_recording(GOLDEN_DIR / "edge_stream.csv")
    dec = core.Decoder(MODEL_TEXT)
    # alternate list rows and NumPy rows: both must be accepted
    got = [d for i, (s, c) in enumerate(zip(rec.signal, rec.stim)) if (d := dec.push(list(s) if i % 2 else s, float(c))) is not None]
    assert dec.samples_seen == len(rec.stim)
    _assert_same(got, core.decode_file(str(GOLDEN_DIR / "model.json"), str(GOLDEN_DIR / "edge_stream.csv")))


def test_wrong_sample_length_is_a_value_error(core):
    with pytest.raises(ValueError, match="length 8"):
        core.Decoder(MODEL_TEXT).push([0.0] * 7, 0.0)


@pytest.mark.parametrize("case", MANIFEST["models"], ids=lambda c: c["file"])
def test_core_rejects_every_invalid_model(core, case):
    with pytest.raises(ValueError, match="invalid model"):
        core.check_model((INVALID_DIR / case["file"]).read_text())


@pytest.mark.parametrize("case", MANIFEST["streams"], ids=lambda c: c["name"])
def test_core_rejects_the_bad_row_without_changing_state(core, case):
    dec = core.Decoder(MODEL_TEXT)
    rows = [[to_float(v) for v in row] for row in case["rows"]]
    for row in rows[: case["bad_row"]]:
        dec.push(row[:8], row[8])
    kind = "stim" if case["kind"] == "stim" else "EEG sample"
    with pytest.raises(ValueError, match=f"invalid {kind} at sample {case['bad_row']}"):
        dec.push(rows[case["bad_row"]][:8], rows[case["bad_row"]][8])
    assert dec.samples_seen == case["bad_row"]


@pytest.mark.parametrize("case", MANIFEST["valid_streams"], ids=lambda c: c["name"])
def test_core_accepts_boundary_codes(core, case):
    dec = core.Decoder(MODEL_TEXT)
    rows = [[to_float(v) for v in row] for row in case["rows"]]
    for row in rows:
        dec.push(row[:8], row[8])
    assert dec.samples_seen == len(rows)


@pytest.mark.parametrize("case", MANIFEST["rail_streams"], ids=lambda c: c["name"])
def test_core_rejected_rows_leave_the_run_counters_alone(core, case):
    dec = core.Decoder((GOLDEN_DIR / "rail_model.json").read_text())
    rejected = {r["row"]: r["kind"] for r in case["rejected"]}
    out = []
    for i, row in enumerate(expand_rows(case["rows"])):
        row = [to_float(v) for v in row]
        if i in rejected:
            seen = dec.samples_seen
            kind = "stim" if rejected[i] == "stim" else "EEG sample"
            with pytest.raises(ValueError, match=f"invalid {kind}"):
                dec.push(row[:8], row[8])
            assert dec.samples_seen == seen
        elif (d := dec.push(row[:8], row[8])) is not None:
            out.append([d["sample"], d["label"], d["pinned"]])
    assert out == case["decisions"]


@pytest.mark.parametrize("case", MANIFEST["valid_models"], ids=lambda c: c["file"])
def test_core_loads_every_valid_rail_abs_spelling(core, case):
    path = INVALID_DIR / case["file"]
    core.check_model(path.read_text())
    got = core.decode_file(str(path), str(GOLDEN_DIR / "rail_stream.csv"))
    expected = json.loads((GOLDEN_DIR / "expected.json").read_text())["rail_stream"]["decisions"]
    if case["rail_abs"] == 1e6:
        _assert_same(got, expected)
    else:
        assert [d["sample"] for d in got] == [e["sample"] for e in expected]
        assert all(d["pinned"] == [] and d["label"] != 0 for d in got)


# Number spellings from the contract's CSV dialect: None = accepted, else the core's error.
# Signed inf/nan parse as numbers and are then refused by the EEG rule.
SPELLINGS = {
    "-131": None, ".5": None, "5.": None, "-.5": None, "+5.": None, "1E2": None, "-1.31e+2": None,
    "-Inf": "invalid EEG sample", "+INFINITY": "invalid EEG sample", "-nan": "invalid EEG sample", "+nan": "invalid EEG sample",
    "1_000": "invalid CSV", "0x10": "invalid CSV", "nan(1)": "invalid CSV", "+-1": "invalid CSV", "1e": "invalid CSV", ".": "invalid CSV",
}


@pytest.mark.parametrize("text", SPELLINGS)
def test_csv_number_spellings_follow_the_dialect(core, tmp_path, text):
    path = tmp_path / "one.csv"
    path.write_text("timestamps,Fz,C3,Cz,C4,Pz,PO7,Oz,PO8,stim\n" + f"0,{text},0,0,0,0,0,0,0,0\n")
    model = str(GOLDEN_DIR / "model.json")
    if SPELLINGS[text] is None:
        load_recording(path)
        assert core.decode_file(model, str(path)) == []
    else:
        with pytest.raises(ValueError):
            load_recording(path)
        with pytest.raises(ValueError, match=SPELLINGS[text]):
            core.decode_file(model, str(path))


@pytest.mark.parametrize("ending", ["\n", "\r\n"])
def test_blank_lines_and_line_endings_follow_the_dialect(core, tmp_path, ending):
    row = "0,1,2,3,4,5,6,7,8,0"
    path = tmp_path / "ok.csv"
    path.write_bytes(ending.join(["timestamps,Fz,C3,Cz,C4,Pz,PO7,Oz,PO8,stim", row, "", " \t ", row, "  "]).encode())
    assert len(load_recording(path).stim) == 2
    assert core.decode_file(str(GOLDEN_DIR / "model.json"), str(path)) == []
    old_mac = tmp_path / "cr.csv"  # CR alone does not end a line
    old_mac.write_bytes("\r".join(["timestamps,Fz,C3,Cz,C4,Pz,PO7,Oz,PO8,stim", row, row]).encode())
    with pytest.raises(ValueError):
        load_recording(old_mac)
    with pytest.raises(ValueError, match="invalid CSV"):
        core.decode_file(str(GOLDEN_DIR / "model.json"), str(old_mac))


def test_overflowing_offsets_take_the_variance_floor_everywhere(core):
    """Finite EEG near +-1e308 overflows once the offset is removed: the window variance is NaN,
    and every implementation must floor it (maxNum semantics) rather than pass the NaN on."""
    n = 1 + 125 + 500 + 3  # one decision: onset at sample 1, window 0.5 s + 2 s at 250 Hz
    signal = np.full((n, 8), 1e308)
    signal[0] = -1e308
    stim = np.zeros(n, dtype=np.int64)
    stim[1] = 131
    with np.errstate(all="ignore"):
        expected = decode_recording(Model.from_json(MODEL_TEXT), Recording(np.arange(n) / 250.0, signal, stim))
    dec = core.Decoder(MODEL_TEXT)
    got = [d for s, c in zip(signal, stim) if (d := dec.push(s, float(c))) is not None]
    assert len(got) == 1 and got[0]["features"] == [pytest.approx(np.log(1e-12))] * len(got[0]["features"])
    _assert_same(got, expected)


def test_decode_file_names_the_file_that_failed(core, tmp_path):
    model, csv = str(GOLDEN_DIR / "model.json"), str(GOLDEN_DIR / "stream.csv")
    bad_model = str(INVALID_DIR / "repeated_key.json")
    cases = [
        ((str(tmp_path / "missing.json"), csv), str(tmp_path / "missing.json")),
        ((bad_model, csv), bad_model),
        ((model, str(tmp_path / "missing.csv")), str(tmp_path / "missing.csv")),
        ((model, str(tmp_path)), str(tmp_path)),  # a directory
    ]
    for args, culprit in cases:
        with pytest.raises(ValueError, match="^" + re.escape(culprit + ": ")):
            core.decode_file(*args)


def test_native_cores_agree_with_each_other_bit_for_bit():
    """Same operations in the same order, no fused multiply-add: Rust and C++ should match exactly."""
    cores = []
    for name in CORES:
        try:
            cores.append(importlib.import_module(name))
        except ImportError:
            if name in REQUIRED:
                raise
    if len(cores) < 2:
        pytest.skip("needs both native cores built")
    for name, model_file in CASES.items():
        args = (str(GOLDEN_DIR / model_file), str(GOLDEN_DIR / f"{name}.csv"))
        rust, cpp = (c.decode_file(*args) for c in cores)
        assert rust == cpp, name
