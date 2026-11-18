"""Fixtures that run a test once per decoder core, skipping a native core that is not built
(failing instead when REQUIRE_CORES names it; see helpers.py)."""
import pytest

from helpers import native_or_none
from mi_ref.cores import CORES, NATIVE


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
