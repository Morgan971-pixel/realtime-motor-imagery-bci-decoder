"""Causal band-pass filter bank. Causal on purpose: a live decoder cannot see the future."""
import numpy as np
from scipy.signal import butter, sosfilt

DEFAULT_BANDS = ((8.0, 12.0), (12.0, 16.0), (16.0, 24.0), (24.0, 30.0))
FILTER_ORDER = 4


def design_band(low_hz: float, high_hz: float, fs: float, order: int = FILTER_ORDER) -> np.ndarray:
    """Butterworth band-pass as second-order sections, shape (order, 6)."""
    return butter(order, [low_hz, high_hz], btype="bandpass", fs=fs, output="sos")


def remove_offset(signal: np.ndarray) -> np.ndarray:
    """Subtract each channel's first sample from a (time, channels) array.

    The band-pass already blocks DC; this shortens the start-up transient and keeps
    values small for low-precision ports. A live decoder latches the stream's first
    sample once and never updates it.
    """
    return signal - signal[0]


def filter_causal(sos: np.ndarray, signal: np.ndarray) -> np.ndarray:
    """Filter a (time, channels) array along time from zero initial state, in float64."""
    return sosfilt(sos, signal, axis=0)


def sosfilt_df2t(sos: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Plain-loop spec of the filter for one channel: transposed direct form II.

    This is the exact recurrence the Rust and C++ cores implement, in this order, in float64.
    x is one channel, shape (n,).
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"sosfilt_df2t takes one channel, shape (n,), got {x.shape}")
    z = np.zeros((sos.shape[0], 2))
    y = np.empty(len(x))
    for n, sample in enumerate(x):
        cur = sample
        for s in range(sos.shape[0]):
            b0, b1, b2, _a0, a1, a2 = sos[s]
            out = b0 * cur + z[s, 0]
            z[s, 0] = b1 * cur - a1 * out + z[s, 1]
            z[s, 1] = b2 * cur - a2 * out
            cur = out
        y[n] = cur
    return y
