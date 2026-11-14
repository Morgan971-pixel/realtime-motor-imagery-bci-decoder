"""Invalid inputs every implementation must reject, written to golden/invalid/.

Built from the committed golden/model.json, so they are synthetic and safe to commit.
Non-finite stream values are written as the strings "nan", "inf" and "-inf" because JSON
has no NaN or Infinity; harnesses convert them back to floats.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

from .golden import GOLDEN_DIR, RAIL_ABS
from .model import Model
from .recording import EEG_CHANNELS
from .streaming import StreamingDecoder

INVALID_DIR = GOLDEN_DIR / "invalid"


def _set(path, value):
    def mutate(d):
        *parents, last = path
        for key in parents:
            d = d[key]
        d[last] = value
    return mutate


def _drop(key):
    def mutate(d):
        del d[key]
    return mutate


# name -> (what is wrong, mutation applied to the parsed golden model)
def _all_bands(key, value):
    def mutate(d):
        for band in d["bands"]:
            band[key] = copy.deepcopy(value)
    return mutate


def _both(*mutations):
    def mutate(d):
        for m in mutations:
            m(d)
    return mutate


def _last(d, key):
    """Path to the last element of the last band's `key` list."""
    last_band = len(d["bands"]) - 1
    return ["bands", last_band, key, len(d["bands"][last_band][key]) - 1]


def _at_last(key, index, value):
    def mutate(d):
        *path, row = _last(d, key)
        _set([*path, row, index], value)(d)
    return mutate


# name -> (what is wrong, mutation applied to the parsed golden model)
MODEL_MUTATIONS = {
    # wrong types for numbers, at every level
    "lda_b_string": ("a number written as a string", _set(["lda", "b"], "0.5")),
    "lda_b_bool": ("a boolean where a number belongs", _set(["lda", "b"], True)),
    "lda_w_cell_string": ("a weight written as a string", _set(["lda", "w", 5], "0.5")),
    "fs_string": ("fs written as a string", _set(["fs"], "250")),
    "low_hz_string": ("band edge written as a string", _set(["bands", 0, "low_hz"], "8")),
    "window_start_string": ("window value written as a string", _set(["window", "start_s"], "0.5")),
    "csp_cell_string": ("a CSP weight written as a string (first band)", _set(["bands", 0, "csp", 0, 0], "1.0")),
    "csp_cell_string_last_band": ("a CSP weight written as a string (last band, last row)", _at_last("csp", 7, "1.0")),
    "sos_cell_string": ("a filter coefficient written as a string", _set(["bands", 0, "sos", 0, 0], "1.0")),
    "schema_version_float": ("schema_version must be the integer 1", _set(["schema_version"], 1.0)),
    "schema_version_bool": ("schema_version must be the integer 1", _set(["schema_version"], True)),
    "schema_version_2": ("unsupported schema_version", _set(["schema_version"], 2)),
    "labels_bool": ("label values must be integers", _set(["labels"], {"score_le_0": True, "score_gt_0": 2})),
    "labels_float": ("label values must be integers", _set(["labels"], {"score_le_0": 1.0, "score_gt_0": 2.0})),
    "labels_swapped": ("LEFT is 1, RIGHT is 2", _set(["labels"], {"score_le_0": 2, "score_gt_0": 1})),
    # wrong containers
    "window_is_list": ("window must be an object", _set(["window"], [0.5, 2.0])),
    "lda_is_list": ("lda must be an object", _set(["lda"], [0.0])),
    "band_is_list": ("each band must be an object", _set(["bands", 0], [8.0, 12.0])),
    "bands_is_object": ("bands must be a list", _set(["bands"], {"low_hz": 8.0})),
    "channels_is_string": ("channels must be a list", _set(["channels"], "Fz,C3,Cz,C4,Pz,PO7,Oz,PO8")),
    "channels_seven": ("exactly 8 channels", _set(["channels"], ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz"])),
    "channels_reversed": ("channel order is fixed", _set(["channels"], ["PO8", "Oz", "PO7", "Pz", "C4", "Cz", "C3", "Fz"])),
    "channels_duplicated": ("channel names must be the fixed list", _set(["channels"], ["Fz"] * 8)),
    # unknown and missing keys, at every level
    "unknown_key": ("unknown top-level key", _set(["extra"], 1)),
    "unknown_key_in_window": ("unknown key inside window", _set(["window", "extra"], 1)),
    "unknown_key_in_band": ("unknown key inside a band", _set(["bands", 0, "extra"], 1)),
    "unknown_key_in_labels": ("unknown key inside labels", _set(["labels", "extra"], 3)),
    "missing_lda": ("missing top-level key", _drop("lda")),
    "missing_schema_version": ("missing top-level key", _drop("schema_version")),
    "missing_lda_b": ("missing key inside lda", lambda d: d["lda"].pop("b")),
    # values and shapes
    "fs_negative": ("fs must be > 0 (this also puts every band above fs/2)", _set(["fs"], -250.0)),
    "offset_mean": ("offset must be first_sample", _set(["offset"], "mean")),
    "bands_empty": ("at least one band (lda.w emptied too, so only this rule breaks)", _both(_set(["bands"], []), _set(["lda", "w"], []))),
    "band_low_zero": ("low_hz must be > 0", _set(["bands", 0, "low_hz"], 0.0)),
    "band_low_above_high": ("low_hz must be < high_hz", _set(["bands", 0, "low_hz"], 13.0)),
    "band_high_at_nyquist": ("high_hz must be below fs/2, not equal", _set(["bands", 0, "high_hz"], 125.0)),
    "band_above_nyquist": ("high_hz must be below fs/2", _set(["bands", 0, "high_hz"], 130.0)),
    "sos_wrong_width": ("filter sections need 6 numbers", _set(["bands", 0, "sos"], [[1.0, 2.0, 3.0]])),
    "sos_empty": ("a band needs at least one filter section", _set(["bands", 0, "sos"], [])),
    "sos_a0_not_one": ("a0 must equal 1 (first band, first section)", _set(["bands", 0, "sos", 0, 3], 2.0)),
    "sos_a0_not_one_last_band": ("a0 must equal 1 (last band, last section)", _at_last("sos", 3, 2.0)),
    "csp_wrong_width": ("CSP rows need 8 columns (every band)", _all_bands("csp", [[0.0] * 5] * 4)),
    "csp_rows_differ": ("every band needs the same CSP row count", _set(["bands", 1, "csp"], [[0.0] * 8] * 2)),
    "csp_odd_rows": ("CSP row count must be even", _both(_all_bands("csp", [[0.0] * 8] * 3), _set(["lda", "w"], [0.0] * 12))),
    "lda_w_too_short": ("lda.w has 3 numbers, needs bands x 2p", _set(["lda", "w"], [0.0, 1.0, 2.0])),
    "window_negative_start": ("window.start_s must be >= 0", _set(["window", "start_s"], -1.0)),
    "window_len_zero": ("window.len_s must be > 0", _set(["window", "len_s"], 0.0)),
    "window_len_negative": ("window.len_s must be > 0", _set(["window", "len_s"], -2.0)),
    "window_too_short": ("window must span at least 2 samples (rounds to 0)", _set(["window", "len_s"], 0.001)),
    "window_one_sample": ("window must span at least 2 samples (rounds to 1)", _set(["window", "len_s"], 0.004)),
    "window_len_huge": ("window length above 2^20 samples", _set(["window", "len_s"], 1e12)),
    "window_start_huge": ("window start above 2^20 samples", _set(["window", "start_s"], 1e12)),
    "fs_huge": ("fs x len_s overflows to infinity", _set(["fs"], 1e308)),
    "channel_not_string": ("channel names must be strings", _set(["channels", 0], 1)),
    "lda_w_too_long": ("lda.w has one number too many", lambda d: d["lda"]["w"].append(0.0)),
    "csp_empty_rows": ("a band needs CSP rows (lda.w emptied too, so only this rule breaks)", _both(_all_bands("csp", []), _set(["lda", "w"], []))),
    # the optional rail_abs: when present, a finite number > 0
    "rail_abs_zero": ("rail_abs must be > 0", _set(["rail_abs"], 0.0)),
    "rail_abs_negative": ("rail_abs must be > 0", _set(["rail_abs"], -700000.0)),
    "rail_abs_string": ("rail_abs written as a string", _set(["rail_abs"], "700000")),
    "rail_abs_null": ("null is not a number (leave the key out to turn the check off)", _set(["rail_abs"], None)),
    "rail_abs_bool": ("a boolean where rail_abs belongs", _set(["rail_abs"], True)),
}

# name -> (what is wrong, function from the golden JSON text to the invalid text)
MODEL_TEXT_EDITS = {
    "repeated_key": ("a key appears twice", lambda t: _insert_once(t, '"fs": ', '"fs": 250.0, "fs": ')),
    "repeated_key_in_window": ("a key appears twice inside window", lambda t: _insert_once(t, '"start_s": ', '"start_s": 0.5, "start_s": ')),
    "nan_literal": ("JSON has no NaN", lambda t: _replace_lda_b(t, "NaN")),
    "number_overflow": ("1e400 overflows to infinity, not a finite number", lambda t: _replace_lda_b(t, "1e400")),
    "integer_overflow": ("a 401-digit integer is not a finite number", lambda t: _replace_lda_b(t, "1" + "0" * 400)),
    "top_level_array": ("the document must be an object", lambda t: "[]"),
    "top_level_null": ("the document must be an object", lambda t: "null"),
    "byte_order_mark": ("a UTF-8 byte-order mark before the document is not JSON", lambda t: "\ufeff" + t),
    "rail_abs_overflow": ("rail_abs 1e400 overflows to infinity", lambda t: _insert_once(t, '"labels": ', '"rail_abs": 1e400,\n "labels": ')),
    "rail_abs_duplicate": ("rail_abs appears twice", lambda t: _insert_once(t, '"labels": ', '"rail_abs": 1e6,\n "rail_abs": 1e6,\n "labels": ')),
    "rail_abs_bigint": ("rail_abs as a 401-digit integer is not a finite number", lambda t: _insert_once(t, '"labels": ', '"rail_abs": 1' + "0" * 400 + ',\n "labels": ')),
}


def _insert_once(text: str, marker: str, replacement: str) -> str:
    assert marker in text, marker
    return text.replace(marker, replacement, 1)


# Which kind of load rule each invalid model breaks, so ports can check they reject it for the
# intended reason: "json" (not valid JSON, or a number no double can hold), "keys" (unknown,
# missing or repeated key), "type" (wrong JSON type), "shape" (wrong count or length),
# "value" (right type and shape, forbidden value).
RULES = ("json", "keys", "type", "shape", "value")
MODEL_RULES = {
    "lda_b_string": "type", "lda_b_bool": "type", "lda_w_cell_string": "type", "fs_string": "type",
    "low_hz_string": "type", "window_start_string": "type", "csp_cell_string": "type",
    "csp_cell_string_last_band": "type", "sos_cell_string": "type", "schema_version_float": "type",
    "schema_version_bool": "type", "labels_bool": "type", "labels_float": "type",
    "window_is_list": "type", "lda_is_list": "type", "band_is_list": "type", "bands_is_object": "type",
    "channels_is_string": "type", "channel_not_string": "type", "top_level_array": "type",
    "top_level_null": "type",
    "unknown_key": "keys", "unknown_key_in_window": "keys", "unknown_key_in_band": "keys",
    "unknown_key_in_labels": "keys", "missing_lda": "keys", "missing_schema_version": "keys",
    "missing_lda_b": "keys", "repeated_key": "keys", "repeated_key_in_window": "keys",
    "sos_wrong_width": "shape", "sos_empty": "shape", "csp_wrong_width": "shape",
    "csp_rows_differ": "shape", "csp_odd_rows": "shape", "csp_empty_rows": "shape",
    "lda_w_too_short": "shape", "lda_w_too_long": "shape", "bands_empty": "shape",
    "schema_version_2": "value", "offset_mean": "value", "labels_swapped": "value",
    "channels_reversed": "value", "channels_duplicated": "value", "channels_seven": "value",
    "fs_negative": "value", "fs_huge": "value", "band_low_zero": "value",
    "band_low_above_high": "value", "band_high_at_nyquist": "value", "band_above_nyquist": "value",
    "sos_a0_not_one": "value", "sos_a0_not_one_last_band": "value",
    "window_negative_start": "value", "window_len_zero": "value", "window_len_negative": "value",
    "window_too_short": "value", "window_one_sample": "value", "window_len_huge": "value",
    "window_start_huge": "value",
    "rail_abs_zero": "value", "rail_abs_negative": "value", "rail_abs_string": "type",
    "rail_abs_null": "type", "rail_abs_bool": "type", "rail_abs_overflow": "json",
    "rail_abs_duplicate": "keys", "rail_abs_bigint": "json",
    "nan_literal": "json", "number_overflow": "json", "integer_overflow": "json", "byte_order_mark": "json",
}


def _replace_lda_b(text: str, literal: str) -> str:
    d = json.loads(text)
    marker = f'"b": {json.dumps(d["lda"]["b"])}'
    assert marker in text
    return text.replace(marker, f'"b": {literal}', 1)


def _good_row(i: int) -> list:
    """A valid row with changing EEG values, so filter state and the latched offset matter."""
    return [float(100 * i + 10 * c + 1) for c in range(8)] + [0.0]


def _rows(n_good: int, bad_sample=None, bad_stim=0.0) -> list:
    rows = [_good_row(i) for i in range(n_good)]
    sample = bad_sample if bad_sample is not None else _good_row(n_good)[:8]
    rows.append([*sample, bad_stim])
    return rows


def _nan_at(channel: int, value: str = "nan") -> list:
    sample = _good_row(0)[:8]
    sample[channel] = value
    return sample


# Each stream is valid up to row `bad_row`, which must be rejected with error kind "stim" or "sample".
STREAMS = {
    "stim_negative": ("stim", _rows(3, bad_stim=-1.0)),
    "stim_fraction": ("stim", _rows(3, bad_stim=131.5)),
    "stim_trial_zero": ("stim", _rows(3, bad_stim=31.0)),
    "stim_99": ("stim", _rows(3, bad_stim=99.0)),
    "stim_too_large": ("stim", _rows(3, bad_stim=2147483648.0)),
    "stim_huge": ("stim", _rows(3, bad_stim=1e300)),
    "stim_nan": ("stim", _rows(3, bad_stim="nan")),
    "stim_inf": ("stim", _rows(3, bad_stim="inf")),
    "stim_negative_inf": ("stim", _rows(3, bad_stim="-inf")),
    "stim_first_row": ("stim", _rows(0, bad_stim=31.0)),
    "eeg_nan": ("sample", _rows(3, bad_sample=_nan_at(2))),
    "eeg_negative_inf": ("sample", _rows(3, bad_sample=_nan_at(7, "-inf"))),
    "eeg_first_row": ("sample", _rows(0, bad_sample=_nan_at(0))),
}

# Streams every implementation must ACCEPT in full: the boundary codes.
VALID_STREAMS = {
    "stim_max_code": [_good_row(0), _good_row(1), [*_good_row(2)[:8], 2147483647.0], _good_row(3)],
    "stim_code_100": [_good_row(0), [*_good_row(1)[:8], 100.0], _good_row(2)],
}


# Valid spellings of rail_abs, written into golden/rail_model.json in place of 1000000.0:
# name -> (why it must load, JSON literal, the double it must load as)
RAIL_ABS_LITERAL = '"rail_abs": 1000000.0'
VALID_MODELS = {
    "valid_rail_abs_integer": ("an integer is a JSON number: pins exactly as 1000000.0", "1000000", RAIL_ABS),
    "valid_rail_abs_exponent": ("exponent notation: pins exactly as 1000000.0", "1e6", RAIL_ABS),
    "valid_rail_abs_u64_max": ("an integer above 2^63 still loads; nothing reaches it", "18446744073709551615", 18446744073709551615.0),
}

# Streams decoded with golden/rail_model.json in which a REJECTED row sits inside a run at the
# rail on C3 (a used channel). The rail model's window is L = 500 samples starting 125 after the
# onset, so the onset at row 1 decides at accepted sample 625.
RAIL_CH = EEG_CHANNELS.index("C3")
RAIL_ONSET, RAIL_L = 1, 500
RAIL_D = RAIL_ONSET + 125 + RAIL_L - 1


def _rail_row(at_rail: bool, stim: float = 0.0, nan_at=None) -> list:
    row = [float(1000 * (c + 1)) for c in range(8)] + [stim]
    if at_rail:
        row[RAIL_CH] = RAIL_ABS
    if nan_at is not None:
        row[nan_at] = "nan"
    return row


def _rail_case(first_at_rail: int, bad_row: list, kind: str) -> dict:
    """Accepted samples 0..RAIL_D, C3 at the rail from `first_at_rail` through RAIL_D; `bad_row`
    (rejected with `kind`) is inserted after accepted sample RAIL_D - RAIL_L // 2, mid-run."""
    after = RAIL_D - RAIL_L // 2
    accepted = [_rail_row(first_at_rail <= i <= RAIL_D, 131.0 if i == RAIL_ONSET else 0.0) for i in range(RAIL_D + 1)]
    rows = [*accepted[: after + 1], bad_row, *accepted[after + 1 :]]
    runs = []  # run-length encoded: [count, row]
    for row in rows:
        if runs and runs[-1][1] == row:
            runs[-1][0] += 1
        else:
            runs.append([1, row])
    return {"rows": runs, "rejected": [{"row": after + 1, "kind": kind}]}


# name -> (what it catches, first accepted sample at the rail, rejected row, its error kind)
RAIL_STREAMS = {
    "rail_run_one_short_plus_rejected_stim": (
        "L - 1 accepted samples at the rail plus a rejected one (bad stim) at the rail: decides",
        RAIL_D - RAIL_L + 2, _rail_row(True, stim=99.0), "stim"),
    "rail_run_one_short_plus_rejected_nan": (
        "L - 1 accepted samples at the rail plus a rejected one (NaN on Fz) at the rail: decides",
        RAIL_D - RAIL_L + 2, _rail_row(True, nan_at=0), "sample"),
    "rail_run_whole_window_with_rejected_dip": (
        "the whole window at the rail with a rejected row (bad stim) below it mid-run: refuses",
        RAIL_D - RAIL_L + 1, _rail_row(False, stim=99.0), "stim"),
}


def expand_rows(runs) -> list:
    """Run-length encoded rows ([count, row] pairs) -> the rows."""
    return [row for count, row in runs for _ in range(count)]


def _reference_decisions(model: Model, rows: list, rejected: dict) -> list:
    """[sample, label, pinned] per decision; each rejected row must raise with its kind."""
    dec = StreamingDecoder(model)
    assert (dec.start, dec.L) == (125, RAIL_L)
    out = []
    for i, row in enumerate(rows):
        values = [to_float(v) for v in row]
        if i in rejected:
            seen = dec._index
            try:
                dec.push(np.array(values[:8]), values[8])
            except ValueError as e:
                assert rejected[i] in str(e), (i, e)
            else:
                raise AssertionError(f"row {i} was accepted")
            assert dec._index == seen
            continue
        if (d := dec.push(np.array(values[:8]), int(values[8]))) is not None:
            out.append([d["sample"], d["label"], d["pinned"]])
    return out


def export_invalid(out_dir: Path = INVALID_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    base_text = (GOLDEN_DIR / "model.json").read_text()
    models = []
    for name, (reason, mutate) in MODEL_MUTATIONS.items():
        d = copy.deepcopy(json.loads(base_text))
        mutate(d)
        (out_dir / f"{name}.json").write_text(json.dumps(d, indent=1))
        models.append({"file": f"{name}.json", "reason": reason, "rule": MODEL_RULES[name]})
    for name, (reason, edit) in MODEL_TEXT_EDITS.items():
        (out_dir / f"{name}.json").write_text(edit(base_text))
        models.append({"file": f"{name}.json", "reason": reason, "rule": MODEL_RULES[name]})
    streams = [
        {"name": name, "kind": kind, "bad_row": len(rows) - 1, "rows": rows}
        for name, (kind, rows) in STREAMS.items()
    ]
    valid = [{"name": name, "rows": rows} for name, rows in VALID_STREAMS.items()]
    rail_text = (GOLDEN_DIR / "rail_model.json").read_text()
    rail_model = Model.from_json(rail_text)
    rail_streams = []
    for name, (reason, first, bad, kind) in RAIL_STREAMS.items():
        case = _rail_case(first, bad, kind)
        rejected = {r["row"]: r["kind"] for r in case["rejected"]}
        decisions = _reference_decisions(rail_model, expand_rows(case["rows"]), rejected)
        rail_streams.append({"name": name, "reason": reason, **case, "decisions": decisions})
    valid_models = []
    for name, (reason, literal, value) in VALID_MODELS.items():
        (out_dir / f"{name}.json").write_text(_insert_once(rail_text, RAIL_ABS_LITERAL, f'"rail_abs": {literal}'))
        valid_models.append({"file": f"{name}.json", "reason": reason, "rail_abs": value})
    manifest = {
        "models": models, "streams": streams, "valid_streams": valid,
        "rail_streams": rail_streams, "valid_models": valid_models,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, allow_nan=False))


def to_float(v) -> float:
    """Manifest value -> float ("nan", "inf", "-inf" stand for non-finite values)."""
    return float(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=INVALID_DIR)
    export_invalid(ap.parse_args().out)


if __name__ == "__main__":
    main()
