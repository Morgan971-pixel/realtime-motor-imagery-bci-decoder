"""Attach cue markers from a separate LSL stream to the EEG samples they belong to.

A cue belongs to the EEG sample nearest to it in time: the first sample whose timestamp is at
least the cue's timestamp minus half a sample period (a cue within floating-point rounding of
exactly halfway may land on either neighbour). Both
timestamps must already be on the local LSL clock (the inlets use clock synchronisation, and the
EEG inlet also smooths jitter). That sample then carries the cue's code as its stim value,
exactly as in a recorded CSV, so the decoder contract does not change.

EEG samples are held back until the newest EEG timestamp is `lag` seconds past them, so a cue
that arrives a little after its sample still lands. A cue whose sample has already been
released is late: it is counted and reported, never moved to a later sample. Invalid codes, a
second cue on an already-cued sample, and cues still waiting when the stream ends are also
counted, reported and dropped.
"""
import bisect
import math
from collections import deque
from collections.abc import Callable

from .events import MAX_CODE


def parse_code(raw) -> int | None:
    """A cue code from an int, an integral float or a string of digits; None if it is not a
    valid stim code (an integer from 100 to 2^31 - 1)."""
    if isinstance(raw, bool):
        return None
    if isinstance(raw, str):
        raw = raw.strip()
        if not (raw.isascii() and raw.isdigit()) or len(raw) > 10:  # int() also reads other scripts' digits
            return None
        code = int(raw)
    elif isinstance(raw, int):
        code = raw
    elif isinstance(raw, float) and math.isfinite(raw) and raw.is_integer():
        code = int(raw)
    else:
        return None
    return code if 100 <= code <= MAX_CODE else None


class CueMerger:
    def __init__(self, fs: float, lag: float = 0.5, report: Callable[[str], None] | None = None):
        if not (math.isfinite(fs) and fs > 0):
            raise ValueError(f"fs must be finite and > 0, got {fs}")
        if not (math.isfinite(lag) and lag >= 0):
            raise ValueError(f"cue lag must be finite and >= 0, got {lag}")
        self.half = 0.5 / fs
        self.lag = lag
        self.report = report or (lambda message: None)
        self._eeg: deque = deque()  # (values, timestamp) not yet released
        self._cues: list[tuple[float, int, int]] = []  # (timestamp, arrival order, code), sorted
        self._arrivals = 0
        self._newest: float | None = None
        self._last_released: float | None = None
        self.late = self.invalid = self.collisions = self.unmatched = 0

    def add_eeg(self, values, timestamp: float) -> None:
        if not math.isfinite(timestamp):
            raise ValueError(f"EEG timestamp must be finite, got {timestamp}")
        if self._newest is not None and timestamp < self._newest:
            raise ValueError(f"EEG timestamp went backwards: {timestamp} after {self._newest}")
        self._eeg.append((values, timestamp))
        self._newest = timestamp

    def add_cue(self, raw, timestamp: float) -> None:
        code = parse_code(raw)
        if code is None or not math.isfinite(timestamp):
            self.invalid += 1
            self.report(f"dropped invalid cue {raw!r} at {timestamp:.6f} s")
            return
        if self._last_released is not None and timestamp - self.half <= self._last_released:
            self.late += 1
            self.report(f"dropped late cue {code} at {timestamp:.6f} s: its EEG sample was already decoded "
                        f"(raise --cue-lag above {self.lag:g} s)")
            return
        bisect.insort(self._cues, (timestamp, self._arrivals, code))
        self._arrivals += 1

    def pop_ready(self) -> list[tuple]:
        """Release every held sample at least `lag` seconds older than the newest EEG sample."""
        out = []
        while self._eeg and self._eeg[0][1] <= self._newest - self.lag:
            out.append(self._release(*self._eeg.popleft()))
        return out

    def flush(self) -> list[tuple]:
        """Release everything still held (end of stream)."""
        out = [self._release(*item) for item in self._eeg]
        self._eeg.clear()
        if self._cues:
            self.unmatched += len(self._cues)
            self.report(f"dropped {len(self._cues)} cue(s) after the last EEG sample: {[c for _, _, c in self._cues]}")
            self._cues.clear()
        return out

    def summary(self) -> str | None:
        """One line counting every dropped cue, or None if none was dropped."""
        if not (self.late or self.invalid or self.collisions or self.unmatched):
            return None
        return (f"cues dropped: {self.late} late, {self.invalid} invalid, {self.collisions} on an already-cued "
                f"sample, {self.unmatched} after the last EEG sample")

    def _release(self, values, timestamp: float) -> tuple:
        stim = 0
        while self._cues and self._cues[0][0] - self.half <= timestamp:
            _, _, code = self._cues.pop(0)
            if stim:
                self.collisions += 1
                self.report(f"dropped cue {code}: cue {stim} is already on the same EEG sample")
                continue
            stim = code
        self._last_released = timestamp
        return values, stim, timestamp
