"""Train the FBCSP + LDA decoder and serialize it to the JSON contract the native cores load."""
import json
from dataclasses import dataclass

import numpy as np

from .csp import fit_csp, log_variance, log_variance_batch
from .epochs import WINDOW_LEN_S, WINDOW_START_S, decision_sample, to_samples, window_slice
from .events import LEFT, RIGHT, imagery_markers, markers_from_stim
from .filters import DEFAULT_BANDS, design_band, filter_causal, remove_offset
from .lda import fit_lda
from .recording import EEG_CHANNELS, FS, Recording

SCHEMA_VERSION = 1
OFFSET_RULE = "first_sample"
MAX_WINDOW_SAMPLES = 2**20  # start and length each at most 2^20 samples (about 70 min at 250 Hz)
LABELS = {"score_le_0": LEFT, "score_gt_0": RIGHT}
_TOP_KEYS = {"schema_version", "fs", "channels", "offset", "window", "bands", "lda", "labels"}
_OPTIONAL_KEYS = {"rail_abs"}  # may be left out; turns on the pinned-channel check
REFUSED = 0  # label of a decision the decoder refused: a used channel sat at the rail all window


def _no_constants(name: str):
    raise ValueError(f"model JSON contains {name}; only finite numbers are allowed")


def _no_duplicates(pairs: list) -> dict:
    keys = [k for k, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError(f"model JSON repeats a key: {sorted(k for k in set(keys) if keys.count(k) > 1)}")
    return dict(pairs)


def _parse_int(text: str):
    """JSON integers stay Python ints, except -0: an int has no negative zero, and the contract reads
    every number as its correctly rounded double, which for -0 is -0.0 (as Rust and C++ do)."""
    return -0.0 if text == "-0" else int(text)


def _num(v, where: str) -> float:
    """A JSON number (int or float, never a bool or a string)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"{where} must be a number, got {v!r}")
    try:
        return float(v)
    except OverflowError:  # a huge integer literal: not a finite number
        raise ValueError(f"{where} is out of range") from None


def _nums(v, where: str, depth: int) -> list:
    """A nested list of JSON numbers, `depth` levels deep."""
    if not isinstance(v, list):
        raise ValueError(f"{where} must be a list, got {type(v).__name__}")
    if depth == 1:
        return [_num(x, where) for x in v]
    return [_nums(x, where, depth - 1) for x in v]


def _keys(d: dict, expected: set, where: str, optional: set = frozenset()) -> dict:
    if not isinstance(d, dict) or not expected <= set(d) <= expected | optional:
        got = sorted(d) if isinstance(d, dict) else type(d).__name__
        raise ValueError(f"{where}: expected keys {sorted(expected)} (optional {sorted(optional)}), got {got}")
    return d


@dataclass
class Model:
    fs: float
    channels: tuple[str, ...]
    window_start_s: float
    window_len_s: float
    bands: tuple[tuple[float, float], ...]
    sos: list[np.ndarray]  # per band, (sections, 6)
    csp: list[np.ndarray]  # per band, (2 * n_pairs, channels)
    lda_w: np.ndarray  # (n_bands * 2 * n_pairs,)
    lda_b: float
    rail_abs: float | None = None  # None: no pinned-channel check

    @property
    def used_channels(self) -> tuple[int, ...]:
        """Channels with a nonzero CSP weight in some band; only these can make a decision refuse."""
        return tuple(c for c in range(len(self.channels)) if any(np.any(W[:, c] != 0.0) for W in self.csp))

    def __post_init__(self) -> None:
        """Reject anything the Rust and C++ loaders would reject."""
        self.channels = tuple(self.channels)
        self.bands = tuple((float(lo), float(hi)) for lo, hi in self.bands)
        self.sos = [np.asarray(x, dtype=np.float64) for x in self.sos]
        self.csp = [np.asarray(x, dtype=np.float64) for x in self.csp]
        self.lda_w = np.asarray(self.lda_w, dtype=np.float64)
        self.fs, self.lda_b = float(self.fs), float(self.lda_b)
        self.window_start_s, self.window_len_s = float(self.window_start_s), float(self.window_len_s)

        def finite(*values) -> bool:
            return all(np.all(np.isfinite(v)) for v in values)

        if not (finite(self.fs) and self.fs > 0):
            raise ValueError(f"fs must be finite and > 0, got {self.fs}")
        if not (finite(self.window_start_s, self.window_len_s) and self.window_start_s >= 0 and self.window_len_s > 0):
            raise ValueError("window needs start_s >= 0 and len_s > 0, both finite")
        for name, seconds in (("start_s", self.window_start_s), ("len_s", self.window_len_s)):
            if not (seconds * self.fs <= MAX_WINDOW_SAMPLES):  # also catches inf from overflow
                raise ValueError(f"window {name} spans more than {MAX_WINDOW_SAMPLES} samples")
        if to_samples(self.window_len_s, self.fs) < 2:
            raise ValueError("window must span at least 2 samples")
        if self.channels != EEG_CHANNELS:
            raise ValueError(f"channels must be {list(EEG_CHANNELS)} in that order, got {list(self.channels)}")
        if not self.bands or not (len(self.bands) == len(self.sos) == len(self.csp)):
            raise ValueError("need at least one band, with one sos and one csp per band")
        for lo, hi in self.bands:
            if not (finite(lo, hi) and 0 < lo < hi < self.fs / 2):
                raise ValueError(f"band ({lo}, {hi}) must satisfy 0 < low < high < fs/2")
        for s in self.sos:
            if s.ndim != 2 or s.shape[0] < 1 or s.shape[1] != 6 or not finite(s) or not np.all(s[:, 3] == 1.0):
                raise ValueError(f"each sos must be (sections, 6), finite, with a0 == 1; got shape {s.shape}")
        rows = {W.shape for W in self.csp}
        if len(rows) != 1 or any(W.ndim != 2 or W.shape[0] % 2 or W.shape[0] == 0 for W in self.csp):
            raise ValueError(f"every csp must share one (2 * n_pairs, channels) shape, got {sorted(rows)}")
        n_rows, n_cols = rows.pop()
        if n_cols != len(EEG_CHANNELS) or not finite(*self.csp):
            raise ValueError(f"csp must have {len(EEG_CHANNELS)} finite columns")
        if self.lda_w.shape != (len(self.bands) * n_rows,) or not finite(self.lda_w, self.lda_b):
            raise ValueError(f"lda.w must be {len(self.bands) * n_rows} finite numbers and lda.b finite")
        if self.rail_abs is not None:
            if isinstance(self.rail_abs, (bool, np.bool_, str)):  # as the loader: never a bool or a string
                raise ValueError(f"rail_abs must be a number, got {self.rail_abs!r}")
            self.rail_abs = float(self.rail_abs)
            if not (finite(self.rail_abs) and self.rail_abs > 0):  # 1e400 reads as inf: refused here
                raise ValueError(f"rail_abs must be finite and > 0, got {self.rail_abs}")

    def to_json(self) -> str:
        rail = {} if self.rail_abs is None else {"rail_abs": self.rail_abs}
        return json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "fs": self.fs,
                "channels": list(self.channels),
                "offset": OFFSET_RULE,
                "window": {"start_s": self.window_start_s, "len_s": self.window_len_s},
                "bands": [
                    {"low_hz": lo, "high_hz": hi, "sos": s.tolist(), "csp": W.tolist()}
                    for (lo, hi), s, W in zip(self.bands, self.sos, self.csp)
                ],
                "lda": {"w": self.lda_w.tolist(), "b": self.lda_b},
                "labels": LABELS,
                **rail,
            },
            indent=1,
            allow_nan=False,  # NaN/Infinity are not JSON; Rust serde_json would reject them
        )

    @classmethod
    def from_json(cls, text: str) -> "Model":
        d = _keys(json.loads(text, parse_int=_parse_int, parse_constant=_no_constants, object_pairs_hook=_no_duplicates), _TOP_KEYS, "model", _OPTIONAL_KEYS)
        if type(d["schema_version"]) is not int or d["schema_version"] != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version {d['schema_version']!r}")
        if d["offset"] != OFFSET_RULE:
            raise ValueError(f"unsupported offset rule {d['offset']!r}")
        if d["labels"] != LABELS or any(type(v) is not int for v in d["labels"].values()):
            raise ValueError(f"labels must be {LABELS}, got {d['labels']}")
        if not isinstance(d["channels"], list) or not all(isinstance(c, str) for c in d["channels"]):
            raise ValueError("channels must be a list of strings")
        window = _keys(d["window"], {"start_s", "len_s"}, "window")
        lda = _keys(d["lda"], {"w", "b"}, "lda")
        if not isinstance(d["bands"], list):
            raise ValueError("bands must be a list")
        bands = [_keys(b, {"low_hz", "high_hz", "sos", "csp"}, f"bands[{i}]") for i, b in enumerate(d["bands"])]
        return cls(
            fs=_num(d["fs"], "fs"),
            channels=d["channels"],
            window_start_s=_num(window["start_s"], "window.start_s"),
            window_len_s=_num(window["len_s"], "window.len_s"),
            bands=[(_num(b["low_hz"], "low_hz"), _num(b["high_hz"], "high_hz")) for b in bands],
            sos=[_nums(b["sos"], f"bands[{i}].sos", 2) for i, b in enumerate(bands)],
            csp=[_nums(b["csp"], f"bands[{i}].csp", 2) for i, b in enumerate(bands)],
            lda_w=_nums(lda["w"], "lda.w", 1),
            lda_b=_num(lda["b"], "lda.b"),
            rail_abs=_num(d["rail_abs"], "rail_abs") if "rail_abs" in d else None,
        )


@dataclass
class EpochSet:
    X: list[np.ndarray]  # per band, (trials, channels, time)
    y: np.ndarray  # (trials,)
    groups: np.ndarray  # (trials,) one id per (recording, trial number); a calibration block shares a number


def filtered_bands(signal: np.ndarray, sos_bank: list[np.ndarray]) -> list[np.ndarray]:
    x = remove_offset(signal)
    return [filter_causal(sos, x) for sos in sos_bank]


def build_epochs(recordings: list[Recording], sos_bank: list[np.ndarray], fs: float = FS,
                 start_s: float = WINDOW_START_S, len_s: float = WINDOW_LEN_S) -> EpochSet:
    """Imagery windows from every recording. A window that runs past the end of its recording
    is skipped, exactly as a live decoder makes no decision if the stream stops early."""
    X: list[list[np.ndarray]] = [[] for _ in sos_bank]
    y, pairs = [], []
    for r_idx, rec in enumerate(recordings):
        bands = filtered_bands(rec.signal, sos_bank)
        for m in imagery_markers(markers_from_stim(rec.stim)):
            sl = window_slice(m.sample, fs, start_s, len_s)
            if sl.stop > len(rec.signal):
                continue
            for b, filt in enumerate(bands):
                X[b].append(filt[sl].T)
            y.append(m.movement)
            pairs.append((r_idx, m.trial))
    if not y:
        raise ValueError("no complete imagery windows in these recordings")
    _, groups = np.unique(np.array(pairs), axis=0, return_inverse=True)  # collision-free ids
    return EpochSet([np.array(xb) for xb in X], np.array(y), groups.ravel())


def features(csp: list[np.ndarray], X: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([log_variance_batch(W, Xb) for W, Xb in zip(csp, X)], axis=1)


def _checked_keep(keep, n_channels: int, n_pairs: int) -> np.ndarray:
    """keep as sorted unique channel indices; refuses anything but integers (booleans, floats and
    strings included, which a cast to int would quietly turn into indices), indices out of range
    and too few channels."""
    items = list(keep)
    # checked one by one: a mixed list such as [True, 0, 2, 3] becomes an integer array as a whole
    if not all(isinstance(k, (int, np.integer)) and not isinstance(k, (bool, np.bool_)) for k in items):
        raise ValueError(f"keep must hold integer channel indices, got {items}")
    keep = np.unique(np.asarray(items, dtype=int))
    if keep.size and (keep[0] < 0 or keep[-1] >= n_channels):
        raise ValueError(f"keep must hold channel indices 0..{n_channels - 1}, got {keep.tolist()}")
    if keep.size < 2 * n_pairs:
        raise ValueError(f"keep needs at least {2 * n_pairs} channels for {n_pairs} CSP pairs, got {keep.size}")
    return keep


def fit_epochs(es: EpochSet, n_pairs: int = 2, csp_alpha: float = 0.1, lda_alpha: float = 0.1,
               keep=None) -> tuple[list[np.ndarray], np.ndarray, float]:
    """keep: channel indices to fit CSP on (default all). Dropped channels get zero CSP columns,
    so they are not used channels: the decoder ignores them, pinned or not."""
    if keep is None:
        csp = [fit_csp(Xb, es.y, n_pairs, csp_alpha) for Xb in es.X]
    else:
        keep = _checked_keep(keep, es.X[0].shape[1], n_pairs)
        csp = []
        for Xb in es.X:
            W = np.zeros((2 * n_pairs, Xb.shape[1]))
            W[:, keep] = fit_csp(Xb[:, keep], es.y, n_pairs, csp_alpha)
            csp.append(W)
    w, b = fit_lda(features(csp, es.X), es.y, lda_alpha)
    return csp, w, b


def train(recordings: list[Recording], bands=DEFAULT_BANDS, fs: float = FS, n_pairs: int = 2,
          csp_alpha: float = 0.1, lda_alpha: float = 0.1, keep=None, rail_abs: float | None = None) -> Model:
    sos_bank = [design_band(lo, hi, fs) for lo, hi in bands]
    start_s, len_s = WINDOW_START_S, WINDOW_LEN_S
    es = build_epochs(recordings, sos_bank, fs, start_s, len_s)
    csp, w, b = fit_epochs(es, n_pairs, csp_alpha, lda_alpha, keep)
    return Model(fs, EEG_CHANNELS, start_s, len_s, tuple(bands), sos_bank, csp, w, b, rail_abs)


def pinned_channels(model: Model, signal: np.ndarray, window: slice) -> list[int]:
    """Used channels whose raw value is at or beyond rail_abs on every sample of the window."""
    if model.rail_abs is None:
        return []
    return [c for c in model.used_channels if np.all(np.abs(signal[window, c]) >= model.rail_abs)]


def decode_recording(model: Model, rec: Recording) -> list[dict]:
    """Every decision a live decoder would make on this recording, in time order.
    Imagery onsets too close to the end for a full window produce no decision. A decision whose
    window has a used channel pinned at the rail is refused: label REFUSED (0), channels in pinned."""
    bands = filtered_bands(rec.signal, model.sos)
    out = []
    for m in imagery_markers(markers_from_stim(rec.stim)):
        sl = window_slice(m.sample, model.fs, model.window_start_s, model.window_len_s)
        if sl.stop > len(rec.signal):
            continue
        f = np.concatenate([log_variance(W, filt[sl].T) for W, filt in zip(model.csp, bands)])
        score = float(f @ model.lda_w + model.lda_b)
        pinned = pinned_channels(model, rec.signal, sl)
        out.append({
            "sample": decision_sample(m.sample, model.fs, model.window_start_s, model.window_len_s),
            "trial": m.trial,
            "true_label": m.movement,
            "features": f.tolist(),
            "score": score,
            "label": REFUSED if pinned else RIGHT if score > 0.0 else LEFT,
            "pinned": pinned,
        })
    return out
