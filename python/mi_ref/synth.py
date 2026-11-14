"""Synthetic recordings that follow the club protocol (safe to publish).

Each trial: prepare (3 s) -> imagery (5 s) -> rest (3 s). Left imagery suppresses
the 10 Hz mu rhythm over C4, right imagery over C3 (event-related desynchronization).
"""
import numpy as np

from .events import LEFT, RIGHT
from .filters import design_band, filter_causal
from .recording import EEG_CHANNELS, FS, Recording

PREPARE_S, IMAGERY_S, REST_S = 3.0, 5.0, 3.0
LEAD_S = 2.0
C3 = EEG_CHANNELS.index("C3")
C4 = EEG_CHANNELS.index("C4")


def make_recording(n_trials: int = 32, seed: int = 0, effect: float = 0.6, fs: float = FS) -> Recording:
    """effect = fraction of mu amplitude removed over the contralateral channel (0 = no signal).

    The default 0.6 (an 84% power drop) is stronger than typical real ERD (20-50%) on purpose:
    golden tests need a clearly decodable signal.

    Golden files depend on the exact random draws: keep the draw order (narrow-band noise,
    white noise, offsets) unchanged, or regenerate and recommit golden/.
    """
    rng = np.random.default_rng(seed)
    n_ch = len(EEG_CHANNELS)
    prep, imag_len = int(PREPARE_S * fs), int(IMAGERY_S * fs)
    trial_len = int((PREPARE_S + IMAGERY_S + REST_S) * fs)
    lead = int(LEAD_S * fs)
    n = 2 * lead + n_trials * trial_len
    t = np.arange(n) / fs

    mu_amp = np.full((n, n_ch), 4.0)
    stim = np.zeros(n, dtype=np.int64)
    for k in range(n_trials):
        trial, movement = k + 1, (LEFT if k % 2 == 0 else RIGHT)
        start = lead + k * trial_len
        imag = start + prep
        rest = imag + imag_len
        stim[start] = trial * 100 + 20 + movement
        stim[imag] = trial * 100 + 30 + movement
        stim[rest] = trial * 100 + 50 + movement
        mu_amp[imag:rest, C4 if movement == LEFT else C3] *= 1.0 - effect

    # Independent 9-11 Hz noise per channel: real mu rhythms are not phase-locked across the scalp.
    narrow = filter_causal(design_band(9.0, 11.0, fs), rng.normal(size=(n, n_ch)))
    mu = mu_amp * narrow / narrow.std(axis=0)
    noise = rng.normal(0.0, 5.0, size=(n, n_ch))
    offsets = rng.uniform(2e5, 8e5, size=n_ch)
    return Recording(timestamps=1.7e9 + t, signal=offsets + mu + noise, stim=stim)
