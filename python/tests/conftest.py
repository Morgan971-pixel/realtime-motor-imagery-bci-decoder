"""Fixtures that run a test once per decoder core, skipping a native core that is not built
(failing instead when REQUIRE_CORES names it; see helpers.py), and the Hypothesis profiles.

Hypothesis profiles (HYPOTHESIS_PROFILE picks one; dev is the default):
- dev: 100 new random examples per property, failures remembered in .hypothesis/ (git-ignored);
- ci: 300 examples, the same ones on every run (derandomize), so a red build always means a change.
Neither has a per-example deadline: timing on a busy machine is not a property of the code.
"""
import os

import pytest
from hypothesis import settings

from helpers import native_or_none
from mi_ref.cores import CORES, NATIVE

settings.register_profile("dev", max_examples=100, deadline=None)
settings.register_profile("ci", max_examples=300, deadline=None, derandomize=True, database=None, print_blob=True)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))


@pytest.fixture(params=tuple(NATIVE))
def native_core(request):
    """The binding module of each native core: mi_core (Rust), then mi_core_cpp (C++)."""
    module = native_or_none(request.param)
    if module is None:
        pytest.skip(f"{NATIVE[request.param]} is not built")
    return module


@pytest.fixture(params=CORES)
def any_core(request):
    """The name of each core (python, rust, cpp), for mi_ref.cores.make_decoder."""
    if request.param in NATIVE and native_or_none(request.param) is None:
        pytest.skip(f"{NATIVE[request.param]} is not built")
    return request.param
