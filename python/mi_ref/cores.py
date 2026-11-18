"""The three decoder cores, and the one place that knows how to load them.

Every core decodes one sample at a time: push(sample, code) returns a decision or None
(docs/model-format.md). Python is the reference; Rust and C++ are native Python bindings,
which exist only once built.
"""
import importlib

from .model import Model
from .streaming import StreamingDecoder

CORES = ("python", "rust", "cpp")
NATIVE = {"rust": "mi_core", "cpp": "mi_core_cpp"}  # native core -> its binding module


class CoreNotBuilt(RuntimeError):
    """A native core whose Python binding is not built (or not importable)."""


def native_module(core):
    """The binding module of a native core; CoreNotBuilt when it is not built."""
    if core not in NATIVE:
        raise ValueError(f"unknown native core {core!r}; choose {' or '.join(NATIVE)}")
    try:
        return importlib.import_module(NATIVE[core])
    except ImportError as exc:
        raise CoreNotBuilt(f"core {core!r} requires {NATIVE[core]}; build its Python binding first") from exc


def make_decoder(core, model_json):
    """A fresh streaming decoder for model_json (the text of a model.json) in the given core."""
    if core == "python":
        return StreamingDecoder(Model.from_json(model_json))
    if core not in NATIVE:
        raise ValueError(f"unknown core {core!r}; choose {', '.join(CORES)}")
    return native_module(core).Decoder(model_json)
