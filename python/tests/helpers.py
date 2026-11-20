"""Shared test plumbing: which native cores must be built, and how decisions are compared.

A native core that is not built is skipped, so the Python suite runs on its own. REQUIRE_CORES
names the bindings that must be there (CI sets REQUIRE_CORES=mi_core,mi_core_cpp); a missing
one is then an error. The fixtures that use this are in conftest.py.
"""
import os
import struct

import numpy as np
import pytest

from mi_ref.cores import NATIVE, CoreNotBuilt, native_module


def _required():
    names = {name.strip() for name in os.environ.get("REQUIRE_CORES", "").split(",") if name.strip()}
    unknown = names - set(NATIVE.values())
    if unknown:
        raise pytest.UsageError(f"unknown core in REQUIRE_CORES: {sorted(unknown)}")
    return {core for core, module in NATIVE.items() if module in names}


REQUIRED = _required()  # native core names ("rust", "cpp") that must be built


def native_or_none(core):
    """The binding module of a native core, None when it is not built (an error if REQUIRED)."""
    try:
        return native_module(core)
    except CoreNotBuilt:
        if core in REQUIRED:
            raise
        return None


def assert_same_decisions(got, expected):
    """Same decisions in the same order; features and score equal up to floating-point noise."""
    keys = ("sample", "trial", "true_label", "label", "pinned")
    assert [tuple(d[k] for k in keys) for d in got] == [tuple(e[k] for k in keys) for e in expected]
    for d, e in zip(got, expected):
        np.testing.assert_allclose(d["features"], e["features"], rtol=1e-9)
        assert abs(d["score"] - e["score"]) < 1e-7


TIE = 1e-9  # a score this close to 0 may round to either label between implementations


def assert_close_decisions(got, expected):
    """For implementations that agree within floating-point noise but not bit for bit (the Python
    reference against a native core, or streaming against batch): the same decisions in the same
    order, features within 1e-9 relative, scores within 1e-7, and the same label unless the score
    is a tie (|score| <= TIE), where rounding may land either side of 0."""
    keys = ("sample", "trial", "true_label", "pinned")
    assert [tuple(d[k] for k in keys) for d in got] == [tuple(e[k] for k in keys) for e in expected]
    for d, e in zip(got, expected):
        np.testing.assert_allclose(d["features"], e["features"], rtol=1e-9)
        assert abs(d["score"] - e["score"]) < 1e-7
        assert d["label"] == e["label"] or (abs(d["score"]) <= TIE and abs(e["score"]) <= TIE), (d, e)


def exact_bits(value):
    """value with every float replaced by its raw IEEE-754 bytes, so 0.0 differs from -0.0 and NaNs
    are equal only with the same sign and payload: the comparison bit-identical cores must pass."""
    if isinstance(value, float):
        return struct.pack("<d", value).hex()
    if isinstance(value, dict):
        return {k: exact_bits(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [exact_bits(v) for v in value]
    return value


def outcome(decoder, signal, stim):
    """Every decision a decoder makes on the stream (as exact_bits), or the error that stopped it."""
    decisions = []
    try:
        for sample, code in zip(signal, stim):
            if (d := decoder.push(sample, float(code))) is not None:
                decisions.append(d)
    except ValueError as err:
        return {"error": str(err).split(":")[0], "after": len(decisions)}
    return exact_bits(decisions)

