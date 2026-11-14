"""Sample-at-a-time decoder that follows docs/model-format.md literally.

Slow on purpose: it is the blueprint for the Rust and C++ cores, not a production path.
"""
from dataclasses import dataclass, field

import numpy as np

from .csp import VAR_FLOOR
from .epochs import to_samples
from .events import IMAGERY_PHASE, LEFT, RIGHT, parse_code, validate_stim
from .model import REFUSED, Model


@dataclass
class StreamingDecoder:
    model: Model
    L: int = field(init=False)  # window length in samples
    start: int = field(init=False)  # window start offset after the onset, in samples
    _x0: np.ndarray | None = field(init=False, default=None)  # latched first sample
    _z: list[np.ndarray] = field(init=False)  # per band: (sections, channels, 2) filter state
    _ring: list[np.ndarray] = field(init=False)  # per band: (L, channels) last L filtered samples
    _index: int = field(init=False, default=-1)
    _last_code: int = field(init=False, default=0)
    _prev_nonzero_at: int = field(init=False, default=-2)
    _pending: dict[int, tuple[int, int]] = field(init=False, default_factory=dict)  # decision sample -> (trial, movement)
    _run: list[int] = field(init=False)  # per channel: consecutive samples at or beyond rail_abs, ending now, at most L
    _used: tuple[int, ...] = field(init=False)  # channels with a nonzero CSP weight

    def __post_init__(self) -> None:
        n_ch = len(self.model.channels)
        self.L = to_samples(self.model.window_len_s, self.model.fs)
        self.start = to_samples(self.model.window_start_s, self.model.fs)
        self._z = [np.zeros((sos.shape[0], n_ch, 2)) for sos in self.model.sos]
        self._ring = [np.zeros((self.L, n_ch)) for _ in self.model.sos]
        self._run = [0] * n_ch
        self._used = self.model.used_channels

    def _filter(self, band: int, x: np.ndarray) -> np.ndarray:
        z, cur = self._z[band], x
        for s, (b0, b1, b2, _a0, a1, a2) in enumerate(self.model.sos[band]):
            out = b0 * cur + z[s, :, 0]
            z[s, :, 0] = b1 * cur - a1 * out + z[s, :, 1]
            z[s, :, 1] = b2 * cur - a2 * out
            cur = out
        return cur

    def push(self, sample, code) -> dict | None:
        """Feed one sample (8 raw values) and its stim code. Returns a decision or None.

        Invalid input raises before any state changes, so the decoder is left as it was.
        """
        sample = np.asarray(sample, dtype=np.float64)
        if sample.shape != (len(self.model.channels),):
            raise ValueError(f"sample must have shape ({len(self.model.channels)},), got {sample.shape}")
        if not np.all(np.isfinite(sample)):
            raise ValueError("sample contains NaN or infinite values")
        code = int(validate_stim(np.array([code], dtype=np.float64))[0])
        marker = parse_code(code) if code > 0 else None  # raises for 0 < code < 100

        self._index += 1
        i = self._index
        if self.model.rail_abs is not None:  # raw values, before the offset is removed
            # saturate at L: "pinned" only needs run >= L, and the counter never grows unbounded
            self._run = [min(r + 1, self.L) if abs(v) >= self.model.rail_abs else 0 for r, v in zip(self._run, sample)]
        if self._x0 is None:
            self._x0 = sample.copy()
        x = sample - self._x0
        for b in range(len(self.model.sos)):
            self._ring[b][i % self.L] = self._filter(b, x)  # ring slot i % L; oldest is (i + 1) % L

        # A marker is a nonzero code that does not repeat the previous sample's code.
        if marker and not (code == self._last_code and self._prev_nonzero_at == i - 1):
            trial, phase, movement = marker
            if phase == IMAGERY_PHASE and movement in (LEFT, RIGHT):
                self._pending[i + self.start + self.L - 1] = (trial, movement)
        if code > 0:
            self._last_code, self._prev_nonzero_at = code, i

        if i not in self._pending:
            return None
        trial, movement = self._pending.pop(i)
        feats = []
        for b, W in enumerate(self.model.csp):
            window = np.roll(self._ring[b], -((i + 1) % self.L), axis=0).T  # channels x time, oldest first
            Z = W @ window
            Zc = Z - Z.mean(axis=1, keepdims=True)  # two-pass variance: mean first, then squares
            feats.extend(np.log(np.fmax((Zc * Zc).sum(axis=1) / self.L, VAR_FLOOR)))
        score = float(np.dot(feats, self.model.lda_w) + self.model.lda_b)
        pinned = [c for c in self._used if self._run[c] >= self.L] if self.model.rail_abs is not None else []
        return {"sample": i, "trial": trial, "true_label": movement, "features": feats,
                "score": score, "label": REFUSED if pinned else RIGHT if score > 0.0 else LEFT,
                "pinned": pinned}
