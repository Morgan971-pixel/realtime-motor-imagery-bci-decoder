"""Cho et al. 2017 (GigaDB 100295) left/right hand motor imagery as club-layout recordings.

Source: Cho H, Ahn M, Ahn S, Kwon M, Jun SC, "EEG datasets for motor imagery brain-computer
interface", GigaScience 6(7):1-8, 2017; data doi 10.5524/100295. Licence: CC0 1.0. 52 people,
BioSemi ActiveTwo, 64 EEG channels at 512 Hz. After a 2 s fixation cross an instruction ("left
hand" or "right hand") is shown for 3 s while the person imagines moving that hand; 100 trials
per hand (120 for s07, s09 and s46).

The files store each trial as a 7 s piece (2 s before to 5 s after the instruction), the left-
and right-hand pieces in two separate arrays, each in time order. trial_sequence.zip gives every
trial's onset sample in the session, which puts the pieces back in session order.

Conversion: re-reference to the average of the 64 EEG channels (BioSemi records without a
reference, so this is the usual first step), keep the 8 channels the club headset has (Fz, C3,
Cz, C4, Pz, PO7, Oz, PO8, in the club order), in microvolts; place the pieces end to end in
session order, with timestamps from their true onsets (so the gaps between pieces show); and
put a one-sample stim pulse at each onset with the club code trial * 100 + 30 + movement (phase 3
= imagery, movement 1 = left, 2 = right, trial counting from 1 in session order).

The files carry no channel names, only electrode positions. The 8 channels are taken from the
standard BioSemi 64 order (ROWS), and every file's positions must agree with it (check_positions).

Layout written by fetch: OUT/subjectNNN/session001/sNN.csv, one recording per person, so
`python -m mi_ref.evaluate --data-dir OUT --fs 512` reads it unchanged. A person whose file is
refused gets sNN.refused (the reason) instead of a CSV, and keeps its download in OUT/raw for
inspection; delete the .refused file to try that person again. Downloads go to OUT/raw.

GigaDB publishes no checksums. The SHA-256 in cho2017_sha256.txt were computed from the files as
first downloaded (2026-11-17); every later download must match them, or the fetch stops.
"""
import argparse
import hashlib
import http.client
import io
import sys
import zipfile
import zlib
from collections.abc import Callable
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.io.matlab import MatReadError

from .eegmmidb import download, parse_subjects, pinned_sha256
from .events import IMAGERY_PHASE, LEFT, RIGHT
from .recording import EEG_CHANNELS, Recording, save_recording

FS = 512.0
SUBJECTS = range(1, 53)
BASE_URL = "https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/10.5524/100001_101000/100295"
SEQUENCES = "trial_sequence.zip"
SHA256_FILE = Path(__file__).with_name("cho2017_sha256.txt")
N_EEG = 64  # EEG rows; rows 64-67 are EMG
PIECE = 3584  # samples per stored trial (7 s)
ONSET = 1023  # index of the onset inside each piece, where imagery_event is 1
# BioSemi 64 order (A1-A32, then B1-B32), as used by the dataset's own loaders
BIOSEMI_64 = (
    "FP1", "AF7", "AF3", "F1", "F3", "F5", "F7", "FT7", "FC5", "FC3", "FC1", "C1", "C3", "C5", "T7", "TP7",
    "CP5", "CP3", "CP1", "P1", "P3", "P5", "P7", "P9", "PO7", "PO3", "O1", "IZ", "OZ", "POZ", "PZ", "CPZ",
    "FPZ", "FP2", "AF8", "AF4", "AFZ", "FZ", "F2", "F4", "F6", "F8", "FT8", "FC6", "FC4", "FC2", "FCZ", "CZ",
    "C2", "C4", "C6", "T8", "TP8", "CP6", "CP4", "CP2", "P2", "P4", "P6", "P8", "P10", "PO8", "PO4", "O2",
)
ROWS = {ch: BIOSEMI_64.index(ch.upper()) for ch in EEG_CHANNELS}  # club channel -> row in the file


def load_eeg(path):
    """The `eeg` struct of one person's .mat file (MATLAB 5 format, read by scipy)."""
    mat = loadmat(path, squeeze_me=True, struct_as_record=False, variable_names=["eeg"])
    if "eeg" not in mat:
        raise ValueError(f"{Path(path).name}: no 'eeg' variable")
    return mat["eeg"]


def check_positions(psenloc) -> None:
    """Refuse a file whose electrode positions do not fit ROWS. Positions are digitised on each
    head (x towards the right ear, y towards the nose, z up), and caps sit shifted or tilted by
    up to about 0.2 of the head radius, so only relations are checked: C3 and PO7 on the left,
    C4 and PO8 on the right, C3 in front of PO7 and C4 in front of PO8, each midline channel
    between C3 and C4, and Fz, Cz, Pz, Oz in front-to-back order."""
    p = np.asarray(psenloc, dtype=np.float64)
    if p.shape != (N_EEG, 3) or not np.all(np.isfinite(p)):
        raise ValueError(f"electrode positions: expected {N_EEG} x 3 finite values, got shape {p.shape}")
    x = {ch: p[r][0] for ch, r in ROWS.items()}
    y = {ch: p[r][1] for ch, r in ROWS.items()}
    midline = ("Fz", "Cz", "Pz", "Oz")
    problems = [f"{ch} is not on the left" for ch in ("C3", "PO7") if x[ch] >= 0]
    problems += [f"{ch} is not on the right" for ch in ("C4", "PO8") if x[ch] <= 0]
    problems += [f"{a} is not in front of {b}" for a, b in (("C3", "PO7"), ("C4", "PO8")) if y[a] <= y[b]]
    problems += [f"{ch} is not between C3 and C4" for ch in midline if not x["C3"] < x[ch] < x["C4"]]
    if not all(y[a] > y[b] for a, b in zip(midline, midline[1:])):
        problems.append("Fz, Cz, Pz, Oz are not in front-to-back order")
    if problems:
        raise ValueError("electrode positions do not match the BioSemi 64 order: " + "; ".join(problems))


def _onsets(values, n: int, hand: str) -> np.ndarray:
    a = np.asarray(values).ravel()
    if (a.shape != (n,) or a.dtype.kind not in "iuf"
            or not np.all(np.isfinite(a)) or not np.array_equal(a, np.rint(a))):
        raise ValueError(f"trial sequence ({hand}): expected {n} whole-number onsets, got {a.dtype} {a.shape}")
    a = a.astype(np.int64)
    if np.any(a < ONSET) or np.any(np.diff(a) <= 0):
        raise ValueError(f"trial sequence ({hand}): onsets must be increasing and leave room for the 2 s before")
    return a


def _pieces(data, n: int, hand: str) -> np.ndarray:
    """(n, PIECE, 8) re-referenced club channels, one row per trial."""
    x = np.asarray(data)
    if x.ndim != 2 or x.shape[0] < N_EEG or x.shape[1] != n * PIECE:
        raise ValueError(f"imagery_{hand}: expected >= {N_EEG} rows x {n * PIECE} samples, got {x.shape}")
    eeg = x[:N_EEG].astype(np.float64)
    if not np.all(np.isfinite(eeg)):
        raise ValueError(f"imagery_{hand}: contains a NaN or infinite EEG sample")
    eeg -= eeg.mean(axis=0)  # common average reference over the 64 EEG channels
    club = eeg[list(ROWS.values())]  # (8, n * PIECE)
    return club.T.reshape(n, PIECE, len(ROWS))


def to_recording(eeg, left_onsets, right_onsets) -> Recording:
    """One person's imagery trials as a single club-layout recording in session order. eeg is the
    file's `eeg` struct (attributes as loaded by scipy.io.loadmat with struct_as_record=False);
    left_onsets and right_onsets are that person's trial_sequence onsets, in samples."""
    if float(eeg.srate) != FS:
        raise ValueError(f"sample rate {eeg.srate} Hz, expected {FS:g}")
    n = int(eeg.n_imagery_trials)
    if n < 1:
        raise ValueError(f"n_imagery_trials is {n}")
    check_positions(eeg.psenloc)
    event = np.asarray(eeg.imagery_event).ravel()
    expected = np.zeros(n * PIECE, dtype=bool)
    expected[ONSET::PIECE] = True
    if event.shape != expected.shape or not np.array_equal(event != 0, expected):
        raise ValueError(f"imagery_event: expected one onset at sample {ONSET} of every {PIECE}-sample piece")
    trials = [(int(s), LEFT, piece) for s, piece in zip(_onsets(left_onsets, n, "left"), _pieces(eeg.imagery_left, n, "left"))]
    trials += [(int(s), RIGHT, piece) for s, piece in zip(_onsets(right_onsets, n, "right"), _pieces(eeg.imagery_right, n, "right"))]
    trials.sort(key=lambda t: t[0])
    starts = np.array([s for s, _, _ in trials])
    if np.any(np.diff(starts) < PIECE):
        k = int(np.flatnonzero(np.diff(starts) < PIECE)[0])
        raise ValueError(f"trials {k + 1} and {k + 2} overlap: onsets {starts[k]} and {starts[k + 1]}")
    signal = np.concatenate([piece for _, _, piece in trials])
    timestamps = np.concatenate([(s - ONSET + np.arange(PIECE)) / FS for s, _, _ in trials])
    stim = np.zeros(len(signal), dtype=np.int64)
    for k, (_, movement, _) in enumerate(trials):
        stim[k * PIECE + ONSET] = (k + 1) * 100 + IMAGERY_PHASE * 10 + movement
    return Recording(timestamps, signal, stim)


def sequences(zip_path: Path, subject: int) -> tuple[np.ndarray, np.ndarray]:
    """(left, right) onsets of one person from trial_sequence.zip (member sN_trial_sequence_v1.mat)."""
    member = f"s{subject}_trial_sequence_v1.mat"
    with zipfile.ZipFile(zip_path) as zf:
        if member not in zf.namelist():
            raise ValueError(f"{SEQUENCES} has no {member}")
        data = zf.read(member)
    seq = loadmat(io.BytesIO(data), squeeze_me=True, struct_as_record=False)["trial_sequence"]
    return seq.imagery_left, seq.imagery_right


# what scipy and the struct accessors raise on a file that is not shaped like this dataset (a short
# or truncated file never gets here: its SHA-256 fails first, and OSError still stops the fetch)
_UNREADABLE = (AttributeError, KeyError, TypeError, IndexError, EOFError, NotImplementedError, MatReadError,
               zlib.error)


def _convert(mat: Path, seq_zip: Path, subject: int) -> Recording:
    """to_recording on one person's files; any sign that a file is not shaped like this dataset
    becomes a ValueError, so fetch refuses that person instead of stopping."""
    try:
        return to_recording(load_eeg(mat), *sequences(seq_zip, subject))
    except _UNREADABLE as err:
        raise ValueError(f"{mat.name}: not readable as this dataset ({type(err).__name__}: {err})") from err


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _checked(key: str, dest: Path, get: Callable[[str, Path], None], sums: dict[str, str]) -> Path:
    """dest, downloaded unless already there, and matching its pinned SHA-256."""
    if key not in sums:
        raise ValueError(f"{key}: no pinned SHA-256")
    if not dest.exists():
        part = dest.with_name(dest.name + ".part")
        get(f"{BASE_URL}/{key}", part)
        part.replace(dest)
    if _sha256(dest) != sums[key]:
        dest.unlink()
        raise ValueError(f"{key}: SHA-256 does not match the pinned value (file deleted)")
    return dest


def fetch(out: Path, subjects: list[int], *, keep_mat: bool = False,
          get: Callable[[str, Path], None] | None = None, sums: dict[str, str] | None = None) -> dict[str, int]:
    """Download, check, convert and (unless keep_mat) delete each person's file; a refused person's
    file is kept. People who already have a CSV or a .refused file are skipped, so an interrupted
    fetch can simply be rerun."""
    sums = pinned_sha256(SHA256_FILE) if sums is None else sums
    get = download if get is None else get  # looked up per call, so tests can replace download
    counts = {"converted": 0, "present": 0, "refused": 0}
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    seq_zip = None
    for s in subjects:
        name = f"s{s:02d}"
        session = out / f"subject{s:03d}" / "session001"
        csv, refused = session / f"{name}.csv", session / f"{name}.refused"
        if csv.exists() or refused.exists():
            counts["present"] += 1
            continue
        seq_zip = seq_zip or _checked(SEQUENCES, raw / SEQUENCES, get, sums)
        mat = _checked(f"mat_data/{name}.mat", raw / f"{name}.mat", get, sums)
        session.mkdir(parents=True, exist_ok=True)
        try:
            rec = _convert(mat, seq_zip, s)
        except ValueError as err:
            refused.write_text(f"{err}\n")
            print(f"{name}: refused: {err}", file=sys.stderr)
            counts["refused"] += 1
            continue  # keep the file for inspection
        tmp = session / f"{name}.csv.part"
        save_recording(tmp, rec)
        tmp.replace(csv)
        counts["converted"] += 1
        if not keep_mat:
            mat.unlink()
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m mi_ref.cho2017", description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True, help="output folder (keep it outside git)")
    ap.add_argument("--subjects", default="1-52", help="e.g. 1-52 (default) or 1,5,7-9")
    ap.add_argument("--keep-mat", action="store_true", help="keep the downloaded .mat files in OUT/raw")
    a = ap.parse_args(argv)
    try:
        subjects = parse_subjects(a.subjects, SUBJECTS)
    except ValueError as err:
        ap.error(str(err))
    try:
        counts = fetch(a.out, subjects, keep_mat=a.keep_mat)
    except (OSError, ValueError, http.client.HTTPException, zipfile.BadZipFile) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    print(f"{counts['converted']} converted, {counts['present']} already present, {counts['refused']} refused")
    return 0


if __name__ == "__main__":
    sys.exit(main())
