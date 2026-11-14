"""Where each decision window sits relative to the imagery onset.

Contract: L = to_samples(len_s, fs); start = onset + to_samples(start_s, fs); the window is
samples [start, start + L); the decision happens on its last sample, start + L - 1.
to_samples rounds half away from zero, which is what Rust f64::round and C++ std::round do
(Python's built-in round() rounds half to even and would disagree at, e.g., 0.5 s * 125 Hz).
"""
import math

WINDOW_START_S = 0.5
WINDOW_LEN_S = 2.0


def to_samples(seconds: float, fs: float) -> int:
    """Seconds to a whole number of samples, rounding half away from zero."""
    if seconds < 0 or fs <= 0:
        raise ValueError(f"need seconds >= 0 and fs > 0, got {seconds}, {fs}")
    return math.floor(seconds * fs + 0.5)


def window_slice(onset: int, fs: float, start_s: float = WINDOW_START_S, len_s: float = WINDOW_LEN_S) -> slice:
    start = onset + to_samples(start_s, fs)
    return slice(start, start + to_samples(len_s, fs))


def decision_sample(onset: int, fs: float, start_s: float = WINDOW_START_S, len_s: float = WINDOW_LEN_S) -> int:
    """Index of the last sample in the window: the earliest moment a live decoder can decide."""
    return window_slice(onset, fs, start_s, len_s).stop - 1
