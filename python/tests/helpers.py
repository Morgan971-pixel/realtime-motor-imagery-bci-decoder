"""Shared test plumbing: which native cores must be built, and how decisions are compared.

A native core that is not built is skipped, so the Python suite runs on its own. REQUIRE_CORES
names the bindings that must be there (CI sets REQUIRE_CORES=mi_core,mi_core_cpp); a missing
one is then an error. The fixtures that use this are in conftest.py.
"""
import os

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
