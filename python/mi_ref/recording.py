"""Load and save recordings in the club's CSV layout."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .events import validate_stim

COLUMNS = ("timestamps", "Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8", "stim")
EEG_CHANNELS = COLUMNS[1:9]
FS = 250.0
_FMT = ["%.6f"] * (len(COLUMNS) - 1) + ["%d"]


@dataclass
class Recording:
    timestamps: np.ndarray  # (n,) float64
    signal: np.ndarray  # (n, 8) float64, raw units as recorded
    stim: np.ndarray  # (n,) integer codes

    def __post_init__(self) -> None:
        self.timestamps = np.asarray(self.timestamps, dtype=np.float64)
        self.signal = np.asarray(self.signal, dtype=np.float64)
        self.stim = np.asarray(self.stim)
        n = len(self.timestamps)
        if self.timestamps.shape != (n,):
            raise ValueError(f"timestamps must be 1-D, got shape {self.timestamps.shape}")
        if self.signal.shape != (n, len(EEG_CHANNELS)):
            raise ValueError(f"signal shape {self.signal.shape} != ({n}, {len(EEG_CHANNELS)})")
        if not np.all(np.isfinite(self.signal)):
            raise ValueError("signal contains a NaN or infinite EEG sample")
        if self.stim.shape != (n,) or self.stim.dtype.kind not in "iu":
            raise ValueError(f"stim must be integer with shape ({n},), got {self.stim.dtype} {self.stim.shape}")


def load_recording(path: str | Path) -> Recording:
    path = Path(path)
    # utf-8-sig drops a byte-order mark; newline="" keeps CR, so only LF and CRLF end a line
    # (as in the native cores), and a line of spaces or tabs counts as blank
    with path.open(encoding="utf-8-sig", newline="") as fh:
        lines = fh.read().split("\n")
    header = tuple(col.strip(" \t\r") for col in lines[0].split(","))
    if header != COLUMNS:
        raise ValueError(f"{path}: unexpected columns {header}")
    body = [line for line in lines[1:] if line.strip(" \t\r")]
    if not body:
        raise ValueError(f"{path}: no samples")
    data = np.loadtxt(body, delimiter=",", ndmin=2, comments=None)
    if data.shape[1] != len(COLUMNS):
        raise ValueError(f"{path}: expected {len(COLUMNS)} columns, got {data.shape[1]}")
    return Recording(
        timestamps=data[:, 0],
        signal=data[:, 1:9].copy(),
        stim=validate_stim(data[:, 9]),  # same strict rules as events.py: no silent rounding
    )


def save_recording(path: str | Path, rec: Recording) -> None:
    data = np.column_stack([rec.timestamps, rec.signal, rec.stim])
    np.savetxt(path, data, delimiter=",", header=",".join(COLUMNS), comments="", fmt=_FMT)
