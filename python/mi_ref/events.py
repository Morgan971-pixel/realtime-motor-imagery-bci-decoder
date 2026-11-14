"""Parse the club's stim-channel event codes.

Code format: trial * 100 + phase * 10 + movement.
Phase 2 = prepare, 3 = motor imagery, 5 = rest (phases 1 and 4 also occur and are ignored).
Movement 1 = left, 2 = right.

Input contract (the Rust and C++ cores enforce the same rules):
- stim is 1-D, finite, non-negative, integer-valued and at most MAX_CODE; anything else
  is an error.
- 0 means no event. Every positive code must have trial >= 1; otherwise it is an error.
- A code repeated on the very next sample extends the same pulse; the marker reports
  the pulse's first sample (its onset). The same code later, after a gap, is a new marker.
"""
from dataclasses import dataclass

import numpy as np

IMAGERY_PHASE = 3
LEFT = 1
RIGHT = 2
MAX_CODE = 2**31 - 1  # fits a signed 32-bit integer in every language


@dataclass(frozen=True)
class Marker:
    sample: int  # onset sample of the pulse
    trial: int  # taken from the code, not counted
    phase: int
    movement: int


def parse_code(code: int) -> tuple[int, int, int]:
    """Split a stim code into (trial, phase, movement)."""
    if code < 100:
        raise ValueError(f"stim code must be >= 100 (trial >= 1), got {code}")
    return code // 100, (code // 10) % 10, code % 10


def validate_stim(stim: np.ndarray) -> np.ndarray:
    """Enforce the input contract and return the codes as int64. The single stim gatekeeper."""
    stim = np.asarray(stim, dtype=np.float64)
    if stim.ndim != 1:
        raise ValueError(f"stim must be 1-D, got shape {stim.shape}")
    if not np.all(np.isfinite(stim)):
        raise ValueError("stim contains NaN or infinite values")
    if np.any(stim < 0):
        raise ValueError("stim contains negative values")
    if np.any(stim > MAX_CODE):
        raise ValueError(f"stim contains values above {MAX_CODE}")
    if np.any(stim != np.floor(stim)):
        raise ValueError("stim contains non-integer values")
    if np.any((stim > 0) & (stim < 100)):
        raise ValueError("stim contains a nonzero code below 100 (trial >= 1 required)")
    return stim.astype(np.int64)


def markers_from_stim(stim: np.ndarray) -> list[Marker]:
    """One Marker per pulse, in time order."""
    codes = validate_stim(stim)
    markers: list[Marker] = []
    last_idx, last_code = None, 0  # previous nonzero sample and its code
    for i in np.flatnonzero(codes):
        code = int(codes[i])
        if last_idx is not None and i == last_idx + 1 and code == last_code:
            last_idx = int(i)
            continue
        trial, phase, movement = parse_code(code)
        markers.append(Marker(int(i), trial, phase, movement))
        last_idx, last_code = int(i), code
    return markers


def imagery_markers(markers: list[Marker]) -> list[Marker]:
    """Keep only motor-imagery onsets with a valid left/right label."""
    return [m for m in markers if m.phase == IMAGERY_PHASE and m.movement in (LEFT, RIGHT)]
