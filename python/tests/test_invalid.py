import json

import numpy as np
import pytest

from mi_ref.golden import GOLDEN_DIR
from mi_ref.invalid import (
    INVALID_DIR, MODEL_MUTATIONS, MODEL_RULES, MODEL_TEXT_EDITS, RULES, expand_rows, export_invalid, to_float,
)
from mi_ref.model import REFUSED, Model, decode_recording
from mi_ref.recording import COLUMNS, Recording, load_recording
from mi_ref.streaming import StreamingDecoder
from mi_ref.synth import make_recording

MANIFEST = json.loads((INVALID_DIR / "manifest.json").read_text())
MODEL = Model.from_json((GOLDEN_DIR / "model.json").read_text())
RAIL_MODEL = Model.from_json((GOLDEN_DIR / "rail_model.json").read_text())


def _rows(case) -> list[list[float]]:
    return [[to_float(v) for v in row] for row in case["rows"]]


def _tail() -> list[tuple[np.ndarray, int]]:
    """A valid continuation with one imagery onset, long enough to produce a decision."""
    rec = make_recording(n_trials=1, seed=7)
    return list(zip(rec.signal, (int(c) for c in rec.stim)))


def _decisions(dec: StreamingDecoder, rows) -> list:
    return [d for s, c in rows if (d := dec.push(np.asarray(s), c)) is not None]


def _write_csv(path, rows):
    lines = [",".join(COLUMNS)]
    lines += [",".join([str(float(i))] + [repr(v) for v in row]) for i, row in enumerate(rows)]
    path.write_text("\n".join(lines) + "\n")


@pytest.mark.parametrize("case", MANIFEST["models"], ids=lambda c: c["file"])
def test_python_rejects_every_invalid_model(case):
    with pytest.raises(ValueError):
        Model.from_json((INVALID_DIR / case["file"]).read_text())


@pytest.mark.parametrize("case", MANIFEST["streams"], ids=lambda c: c["name"])
def test_streaming_rejects_the_bad_row_and_leaves_no_trace(case):
    rows = _rows(case)
    good, bad = rows[: case["bad_row"]], rows[case["bad_row"]]
    dec = StreamingDecoder(MODEL)
    for row in good:
        dec.push(np.array(row[:8]), row[8])
    with pytest.raises(ValueError, match=case["kind"]):
        dec.push(np.array(bad[:8]), bad[8])
    # After the rejection the decoder must behave exactly like one that never saw the bad row.
    clean = StreamingDecoder(MODEL)
    for row in good:
        clean.push(np.array(row[:8]), row[8])
    tail = _tail()
    assert _decisions(dec, tail) == _decisions(clean, tail) != []


@pytest.mark.parametrize("case", MANIFEST["streams"], ids=lambda c: c["name"])
def test_loading_rejects_the_whole_stream(tmp_path, case):
    path = tmp_path / "bad.csv"
    _write_csv(path, _rows(case))
    with pytest.raises(ValueError, match=case["kind"]):
        load_recording(path)


@pytest.mark.parametrize("case", MANIFEST["valid_streams"], ids=lambda c: c["name"])
def test_boundary_codes_are_accepted(tmp_path, case):
    rows = _rows(case)
    dec = StreamingDecoder(MODEL)
    for row in rows:
        dec.push(np.array(row[:8]), row[8])
    path = tmp_path / "ok.csv"
    _write_csv(path, rows)
    decode_recording(MODEL, load_recording(path))


def test_committed_fixtures_match_a_fresh_export(tmp_path):
    export_invalid(tmp_path)
    for f in sorted(INVALID_DIR.iterdir()):
        assert (tmp_path / f.name).read_bytes() == f.read_bytes(), f.name


def test_every_invalid_model_has_exactly_one_known_rule_tag():
    assert set(MODEL_RULES) == set(MODEL_MUTATIONS) | set(MODEL_TEXT_EDITS)
    assert all(case["rule"] in RULES for case in MANIFEST["models"])


@pytest.mark.parametrize("case", MANIFEST["rail_streams"], ids=lambda c: c["name"])
def test_rejected_rows_inside_a_rail_run_leave_the_counters_alone(case):
    rows = [[to_float(v) for v in row] for row in expand_rows(case["rows"])]
    rejected = {r["row"]: r["kind"] for r in case["rejected"]}
    dec, out, accepted = StreamingDecoder(RAIL_MODEL), [], []
    for i, row in enumerate(rows):
        if i in rejected:
            seen = dec._index
            with pytest.raises(ValueError, match=rejected[i]):
                dec.push(np.array(row[:8]), row[8])
            assert dec._index == seen
            continue
        accepted.append(row)
        if (d := dec.push(np.array(row[:8]), row[8])) is not None:
            out.append(d)
    assert [[d["sample"], d["label"], d["pinned"]] for d in out] == case["decisions"]
    assert [d["label"] == REFUSED for d in out] == [bool(d["pinned"]) for d in out]
    # the batch path, given only the accepted rows, agrees
    a = np.array(accepted)
    batch = decode_recording(RAIL_MODEL, Recording(np.arange(len(a)) / 250.0, a[:, :8], a[:, 8].astype(np.int64)))
    assert [[d["sample"], d["label"], d["pinned"]] for d in batch] == case["decisions"]


def test_rail_streams_cover_decide_and_refuse():
    outcomes = {c["name"]: [d[2] for d in c["decisions"]] for c in MANIFEST["rail_streams"]}
    assert sorted(outcomes.values()) == [[[]], [[]], [[1]]]
    assert {r["kind"] for c in MANIFEST["rail_streams"] for r in c["rejected"]} == {"stim", "sample"}


@pytest.mark.parametrize("case", MANIFEST["valid_models"], ids=lambda c: c["file"])
def test_every_valid_rail_abs_spelling_loads_and_decodes(case):
    model = Model.from_json((INVALID_DIR / case["file"]).read_text())
    assert model.rail_abs == case["rail_abs"]
    got = [(d["sample"], d["label"], d["pinned"]) for d in decode_recording(model, load_recording(GOLDEN_DIR / "rail_stream.csv"))]
    expected = json.loads((GOLDEN_DIR / "expected.json").read_text())["rail_stream"]["decisions"]
    if case["rail_abs"] == RAIL_MODEL.rail_abs:  # the same double: the same decisions
        assert got == [(d["sample"], d["label"], d["pinned"]) for d in expected]
    else:  # above every value of the stream: nothing pins
        assert [g[0] for g in got] == [d["sample"] for d in expected]
        assert all(p == [] and label != REFUSED for _, label, p in got)
