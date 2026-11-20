"""Every native core (Rust `mi_core`, C++ `mi_core_cpp`) must agree with the Python reference.

A core that is not built is skipped, so the Python suite runs on its own. Set
REQUIRE_CORES=mi_core,mi_core_cpp (for example in CI) to make a missing build an error.
"""
import json
import math
import re

import numpy as np
import pytest
from hypothesis import event, given
from hypothesis import strategies as st

from helpers import assert_close_decisions, assert_same_decisions, exact_bits, native_or_none, outcome
from mi_ref.cores import NATIVE, make_decoder
from mi_ref.golden import CASES, GOLDEN_DIR
from mi_ref.invalid import INVALID_DIR, expand_rows, to_float
from mi_ref.model import Model, decode_recording
from mi_ref.recording import Recording, load_recording
from mi_ref.streaming import StreamingDecoder
from strategies import models, streams

MODEL_TEXT = (GOLDEN_DIR / "model.json").read_text()
MANIFEST = json.loads((INVALID_DIR / "manifest.json").read_text())


@pytest.mark.parametrize("name", CASES)
def test_core_reproduces_committed_goldens(native_core, name):
    expected = json.loads((GOLDEN_DIR / "expected.json").read_text())[name]["decisions"]
    assert_same_decisions(native_core.decode_file(str(GOLDEN_DIR / CASES[name]), str(GOLDEN_DIR / f"{name}.csv")), expected)


def test_streaming_binding_refuses_pinned_windows(native_core):
    rec = load_recording(GOLDEN_DIR / "rail_stream.csv")
    dec = native_core.Decoder((GOLDEN_DIR / "rail_model.json").read_text())
    got = [d for s, c in zip(rec.signal, rec.stim) if (d := dec.push(s, float(c))) is not None]
    assert [(d["sample"], d["label"], d["pinned"]) for d in got if d["pinned"]] == [
        (1874, 0, [1]), (4624, 0, [2]), (12874, 0, [7]), (15624, 0, [1, 5]), (21124, 0, [0]),
    ]
    assert [d["label"] == 0 for d in got] == [bool(d["pinned"]) for d in got]
    assert_same_decisions(got, native_core.decode_file(str(GOLDEN_DIR / "rail_model.json"), str(GOLDEN_DIR / "rail_stream.csv")))


def test_streaming_binding_matches_batch(native_core):
    rec = load_recording(GOLDEN_DIR / "edge_stream.csv")
    dec = native_core.Decoder(MODEL_TEXT)
    # alternate list rows and NumPy rows: both must be accepted
    got = [d for i, (s, c) in enumerate(zip(rec.signal, rec.stim)) if (d := dec.push(list(s) if i % 2 else s, float(c))) is not None]
    assert dec.samples_seen == len(rec.stim)
    assert_same_decisions(got, native_core.decode_file(str(GOLDEN_DIR / "model.json"), str(GOLDEN_DIR / "edge_stream.csv")))


def test_wrong_sample_length_is_a_value_error(native_core):
    with pytest.raises(ValueError, match="length 8"):
        native_core.Decoder(MODEL_TEXT).push([0.0] * 7, 0.0)


@pytest.mark.parametrize("case", MANIFEST["models"], ids=lambda c: c["file"])
def test_core_rejects_every_invalid_model(native_core, case):
    with pytest.raises(ValueError, match="invalid model"):
        native_core.check_model((INVALID_DIR / case["file"]).read_text())


@pytest.mark.parametrize("case", MANIFEST["streams"], ids=lambda c: c["name"])
def test_core_rejects_the_bad_row_without_changing_state(native_core, case):
    dec = native_core.Decoder(MODEL_TEXT)
    rows = [[to_float(v) for v in row] for row in case["rows"]]
    for row in rows[: case["bad_row"]]:
        dec.push(row[:8], row[8])
    kind = "stim" if case["kind"] == "stim" else "EEG sample"
    with pytest.raises(ValueError, match=f"invalid {kind} at sample {case['bad_row']}"):
        dec.push(rows[case["bad_row"]][:8], rows[case["bad_row"]][8])
    assert dec.samples_seen == case["bad_row"]


@pytest.mark.parametrize("case", MANIFEST["valid_streams"], ids=lambda c: c["name"])
def test_core_accepts_boundary_codes(native_core, case):
    dec = native_core.Decoder(MODEL_TEXT)
    rows = [[to_float(v) for v in row] for row in case["rows"]]
    for row in rows:
        dec.push(row[:8], row[8])
    assert dec.samples_seen == len(rows)


@pytest.mark.parametrize("case", MANIFEST["rail_streams"], ids=lambda c: c["name"])
def test_core_rejected_rows_leave_the_run_counters_alone(native_core, case):
    dec = native_core.Decoder((GOLDEN_DIR / "rail_model.json").read_text())
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
def test_core_loads_every_valid_rail_abs_spelling(native_core, case):
    path = INVALID_DIR / case["file"]
    native_core.check_model(path.read_text())
    got = native_core.decode_file(str(path), str(GOLDEN_DIR / "rail_stream.csv"))
    expected = json.loads((GOLDEN_DIR / "expected.json").read_text())["rail_stream"]["decisions"]
    if case["rail_abs"] == 1e6:
        assert_same_decisions(got, expected)
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
def test_csv_number_spellings_follow_the_dialect(native_core, tmp_path, text):
    path = tmp_path / "one.csv"
    path.write_text("timestamps,Fz,C3,Cz,C4,Pz,PO7,Oz,PO8,stim\n" + f"0,{text},0,0,0,0,0,0,0,0\n")
    model = str(GOLDEN_DIR / "model.json")
    if SPELLINGS[text] is None:
        load_recording(path)
        assert native_core.decode_file(model, str(path)) == []
    else:
        with pytest.raises(ValueError):
            load_recording(path)
        with pytest.raises(ValueError, match=SPELLINGS[text]):
            native_core.decode_file(model, str(path))


@pytest.mark.parametrize("ending", ["\n", "\r\n"])
def test_blank_lines_and_line_endings_follow_the_dialect(native_core, tmp_path, ending):
    row = "0,1,2,3,4,5,6,7,8,0"
    path = tmp_path / "ok.csv"
    path.write_bytes(ending.join(["timestamps,Fz,C3,Cz,C4,Pz,PO7,Oz,PO8,stim", row, "", " \t ", row, "  "]).encode())
    assert len(load_recording(path).stim) == 2
    assert native_core.decode_file(str(GOLDEN_DIR / "model.json"), str(path)) == []
    old_mac = tmp_path / "cr.csv"  # CR alone does not end a line
    old_mac.write_bytes("\r".join(["timestamps,Fz,C3,Cz,C4,Pz,PO7,Oz,PO8,stim", row, row]).encode())
    with pytest.raises(ValueError):
        load_recording(old_mac)
    with pytest.raises(ValueError, match="invalid CSV"):
        native_core.decode_file(str(GOLDEN_DIR / "model.json"), str(old_mac))


def test_overflowing_offsets_take_the_variance_floor_everywhere(native_core):
    """Finite EEG near +-1e308 overflows once the offset is removed: the window variance is NaN,
    and every implementation must floor it (maxNum semantics) rather than pass the NaN on."""
    n = 1 + 125 + 500 + 3  # one decision: onset at sample 1, window 0.5 s + 2 s at 250 Hz
    signal = np.full((n, 8), 1e308)
    signal[0] = -1e308
    stim = np.zeros(n, dtype=np.int64)
    stim[1] = 131
    with np.errstate(all="ignore"):
        expected = decode_recording(Model.from_json(MODEL_TEXT), Recording(np.arange(n) / 250.0, signal, stim))
    dec = native_core.Decoder(MODEL_TEXT)
    got = [d for s, c in zip(signal, stim) if (d := dec.push(s, float(c))) is not None]
    assert len(got) == 1 and got[0]["features"] == [pytest.approx(np.log(1e-12))] * len(got[0]["features"])
    assert_same_decisions(got, expected)


def test_decode_file_names_the_file_that_failed(native_core, tmp_path):
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
            native_core.decode_file(*args)


def test_native_cores_agree_with_each_other_bit_for_bit():
    """Same operations in the same order, no fused multiply-add: Rust and C++ should match exactly."""
    cores = [module for module in map(native_or_none, NATIVE) if module is not None]
    if len(cores) < 2:
        pytest.skip("needs both native cores built")
    for name, model_file in CASES.items():
        args = (str(GOLDEN_DIR / model_file), str(GOLDEN_DIR / f"{name}.csv"))
        rust, cpp = (c.decode_file(*args) for c in cores)
        assert rust == cpp, name


def _decisions(decoder, signal, stim):
    """Every decision a decoder makes on the stream, in order."""
    return [d for sample, code in zip(signal, stim) if (d := decoder.push(sample, float(code))) is not None]


def _tag(decisions):
    """Coverage labels (shown with --hypothesis-show-statistics), so a generator change that stops
    producing refusals or non-finite scores shows up instead of passing quietly."""
    if not decisions:
        event("decisions: none")
        return
    event("decisions: all refused (pinned)" if all(d["pinned"] for d in decisions) else "decisions: some unpinned")
    event("score: some non-finite" if any(not math.isfinite(d["score"]) for d in decisions) else "score: all finite")


def _both_native():
    cores = [native_or_none(core) for core in NATIVE]
    if None in cores:
        pytest.skip("needs both native cores built")
    return cores


@given(st.data())
def test_rust_and_cpp_decide_bit_identically_on_generated_models_and_streams(data):
    """Any valid model, any values (extremes included): Rust and C++ make the same decisions, bit for bit."""
    rust, cpp = _both_native()
    model = data.draw(models(realistic=False), label="model")
    signal, stim = data.draw(streams(model, realistic=False), label="stream")
    text = model.to_json()
    _tag(_decisions(rust.Decoder(text), signal, stim))
    assert outcome(rust.Decoder(text), signal, stim) == outcome(cpp.Decoder(text), signal, stim)


def _decisions_rejecting(decoder, signal, stim, bad):
    """Every decision a decoder makes on the stream; each row whose index is in bad must raise
    ValueError (and is then skipped), every other row must be accepted."""
    decisions = []
    for i, (sample, code) in enumerate(zip(signal, stim)):
        if i in bad:
            with pytest.raises(ValueError):
                decoder.push(sample, float(code))
        elif (d := decoder.push(sample, float(code))) is not None:
            decisions.append(d)
    return decisions


def _without(signal, stim, bad):
    keep = np.setdiff1d(np.arange(len(stim)), bad)
    return signal[keep], stim[keep]


@given(st.data())
def test_rejected_rows_leave_every_decoder_as_if_they_never_arrived(data):
    """Bad rows (invalid stim, non-finite EEG) inside realistic streams: each one raises, and every
    core then decides exactly as it does on the same stream with those rows deleted. Python and the
    native cores agree within tolerance; Rust and C++ bit for bit."""
    cores = ["python"] + [core for core in NATIVE if native_or_none(core) is not None]
    model = data.draw(models(realistic=True), label="model")
    signal, stim, bad = data.draw(streams(model, bad_rows=True), label="stream")
    clean = _without(signal, stim, bad)
    text = model.to_json()
    got = {}
    for core in cores:
        got[core] = _decisions_rejecting(make_decoder(core, text), signal, stim, bad)
        assert exact_bits(got[core]) == exact_bits(_decisions_rejecting(make_decoder(core, text), *clean, ())), core
    for core in NATIVE.keys() & got.keys():
        assert_close_decisions(got[core], got["python"])
    if NATIVE.keys() <= got.keys():
        assert exact_bits(got["rust"]) == exact_bits(got["cpp"])


@given(st.data())
def test_rust_and_cpp_reject_the_same_rows_and_agree_bit_for_bit_on_extreme_streams(data):
    """Any valid model, extreme values, bad rows inserted: both native cores reject every bad row,
    decide as on the stream without them, and agree with each other bit for bit."""
    rust, cpp = _both_native()
    model = data.draw(models(realistic=False), label="model")
    signal, stim, bad = data.draw(streams(model, realistic=False, bad_rows=True), label="stream")
    text = model.to_json()
    got = [exact_bits(_decisions_rejecting(core.Decoder(text), signal, stim, bad)) for core in (rust, cpp)]
    assert got[0] == exact_bits(_decisions_rejecting(rust.Decoder(text), *_without(signal, stim, bad), ()))
    assert got[0] == got[1]


def _built_natives():
    """The native cores that are built (an error for a missing one named in REQUIRE_CORES)."""
    cores = [module for module in map(native_or_none, NATIVE) if module is not None]
    if not cores:
        pytest.skip("no native core is built")
    return cores


@given(st.data())
def test_python_matches_each_native_core_on_generated_models_and_streams(data):
    """Any realistic model and stream: every native core decides as the Python reference does,
    within floating-point noise."""
    natives = _built_natives()
    model = data.draw(models(), label="model")
    signal, stim = data.draw(streams(model), label="stream")
    text = model.to_json()
    expected = _decisions(StreamingDecoder(Model.from_json(text)), signal, stim)
    _tag(expected)
    for core in natives:
        assert_close_decisions(_decisions(core.Decoder(text), signal, stim), expected)


@given(st.data())
def test_golden_model_on_generated_streams_matches_python_and_across_cores(data):
    """The committed golden model on any realistic stream: every native core decides as the
    Python reference does, and Rust and C++ agree bit for bit."""
    natives = _built_natives()
    signal, stim = data.draw(streams(Model.from_json(MODEL_TEXT)), label="stream")
    expected = _decisions(StreamingDecoder(Model.from_json(MODEL_TEXT)), signal, stim)
    for core in natives:
        assert_close_decisions(_decisions(core.Decoder(MODEL_TEXT), signal, stim), expected)
    if len(natives) == 2:
        rust, cpp = natives
        assert outcome(rust.Decoder(MODEL_TEXT), signal, stim) == outcome(cpp.Decoder(MODEL_TEXT), signal, stim)


# Found by test_native_core_reads_any_spelling_of_the_same_numbers (test_model.py): a model whose
# bias is written as the JSON integer -0 made Rust and C++ disagree in the sign of a zero score,
# because C++ (and Python) read -0 as +0. The contract reads -0 as -0.0 in every core.
NEGATIVE_ZERO_MODEL = {
    "schema_version": 1, "fs": 50.0, "channels": ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"],
    "offset": "first_sample", "window": {"start_s": 0.0, "len_s": 0.04},
    "bands": [{"low_hz": 5.0, "high_hz": 10.0,
               "sos": [[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]],
               "csp": [[1.0, 0, 0, 0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0, 0, 0, 0]]}],
    "lda": {"w": [0.0, 0.0], "b": "B"}, "labels": {"score_le_0": 1, "score_gt_0": 2},
}


def test_negative_zero_written_as_an_integer_reads_the_same_in_every_core():
    text = json.dumps(NEGATIVE_ZERO_MODEL).replace('"B"', "-0")
    assert np.signbit(Model.from_json(text).lda_b)
    signal, stim = np.zeros((2, 8)), [131, 0]  # a flat window: every feature at the floor, score -0.0
    rust, cpp = _both_native()
    got = [outcome(core.Decoder(text), signal, stim) for core in (rust, cpp)]
    assert got[0] == got[1]
    assert got[0][0]["score"] == exact_bits(-0.0)

