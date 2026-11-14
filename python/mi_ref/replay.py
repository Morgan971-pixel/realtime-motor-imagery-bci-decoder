"""Replay a recording as if a headset were streaming it, and time every decision.

The same loop exists in the Rust and C++ cores (docs/replay.md):

    start = now
    for each chunk of `chunk` samples (the recording played `repeat` times back to back):
        paced only: sleep until start + (samples up to the end of this chunk) / fs
        handover = now
        push every sample of the chunk; for each decision, latency = now - handover

Fast mode keeps the decoder busy; paced mode lets it idle between chunks like a live
decoder, so caches go cold. Latency starts when the loop wakes up, because a sleeping
thread can wake milliseconds late (macOS timer coalescing), which a decoder woken by real
headset data does not pay.
"""
import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from .model import Model
from .recording import load_recording
from .streaming import StreamingDecoder


class Clock(Protocol):
    def now_ns(self) -> int: ...
    def sleep_until_ns(self, target: int) -> None: ...


class SystemClock:
    """Monotonic nanoseconds since the clock was created."""

    def __init__(self) -> None:
        self._origin = time.perf_counter_ns()

    def now_ns(self) -> int:
        return time.perf_counter_ns() - self._origin

    def sleep_until_ns(self, target: int) -> None:
        delay = target - self.now_ns()
        if delay > 0:
            time.sleep(delay / 1e9)


@dataclass
class ReplayResult:
    decisions: list[dict]
    latencies_ns: list[int]  # one per decision
    samples: int
    elapsed_ns: int


def replay(push: Callable, signal: np.ndarray, stim: np.ndarray, *, fs: float, chunk: int = 25,
           paced: bool = False, repeat: int = 1, clock: Clock | None = None) -> ReplayResult:
    """Feed `signal`/`stim` to `push(sample, code)` chunk by chunk; see the module docstring."""
    if chunk < 1 or repeat < 1:
        raise ValueError(f"chunk and repeat must be at least 1, got chunk={chunk}, repeat={repeat}")
    clock = clock or SystemClock()
    n = len(stim)
    total = n * repeat
    decisions, latencies = [], []
    start = clock.now_ns()
    for first in range(0, total, chunk):
        last = min(first + chunk, total)
        if paced:
            clock.sleep_until_ns(start + int(last * 1e9 / fs))
        handover = clock.now_ns()
        for j in range(first, last):
            d = push(signal[j % n], stim[j % n])
            if d is not None:
                latencies.append(clock.now_ns() - handover)
                decisions.append(d)
    return ReplayResult(decisions, latencies, total, clock.now_ns() - start)


def summary(impl: str, result: ReplayResult, chunk: int, paced: bool, repeat: int) -> dict:
    """The JSON line every implementation's replay command prints."""
    return {
        "impl": impl, "chunk": chunk, "paced": paced, "repeat": repeat,
        "samples": result.samples, "elapsed_ns": result.elapsed_ns,
        "decisions": [[d["sample"], d["label"]] for d in result.decisions],
        "latencies_ns": result.latencies_ns,
    }


def _count(text: str) -> int:
    """A count per docs/replay.md: ASCII decimal digits, from 1 to 2^64 - 1."""
    if not re.fullmatch(r"[0-9]+", text) or not 1 <= int(text) < 2**64:
        raise argparse.ArgumentTypeError(f"expected a whole number from 1 to 2^64 - 1, got {text!r}")
    return int(text)


def _fail(path: str, error: Exception) -> int:
    """Report a rejected file as `PATH: reason`, without repeating a path the reason already starts with."""
    reason = error.strerror if isinstance(error, OSError) and error.strerror else str(error)
    print(f"{path}: {reason.removeprefix(f'{Path(path)}: ')}", file=sys.stderr)
    return 1


def _check_grammar(p: argparse.ArgumentParser, argv: list[str]) -> None:
    """Hold argparse to docs/replay.md, scanning left to right like the Rust and C++ commands:
    argparse alone would accept `--` and negative-number-like positionals, and honor `-h`
    after an unknown option."""
    it = iter(argv)
    for arg in it:
        flag, eq, _ = arg.partition("=") if arg.startswith("--") else (arg, "", "")
        if flag in ("--chunk", "--repeat"):
            if not eq:
                next(it, None)  # the value; argparse checks it
        elif flag in ("-h", "--help") and not eq:
            return
        elif flag == "--paced" and not eq:
            continue
        elif arg.startswith("-") and len(arg) > 1:
            p.error(f"unrecognized argument: {arg}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m mi_ref.replay", description=__doc__.splitlines()[0], allow_abbrev=False)
    p.add_argument("model")
    p.add_argument("csv")
    p.add_argument("--chunk", type=_count, default=25, help="samples per chunk (default 25)")
    p.add_argument("--repeat", type=_count, default=1, help="play the recording this many times (default 1)")
    p.add_argument("--paced", action="store_true", help="deliver chunks at the model's sampling rate")
    _check_grammar(p, sys.argv[1:] if argv is None else argv)
    a = p.parse_args(argv)
    try:
        with open(a.model) as fh:
            model = Model.from_json(fh.read())
    except (OSError, ValueError) as e:
        return _fail(a.model, e)
    try:
        rec = load_recording(a.csv)
    except (OSError, ValueError) as e:
        return _fail(a.csv, e)
    result = replay(StreamingDecoder(model).push, rec.signal, rec.stim, fs=model.fs,
                    chunk=a.chunk, paced=a.paced, repeat=a.repeat)
    json.dump(summary("python", result, a.chunk, a.paced, a.repeat), sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
