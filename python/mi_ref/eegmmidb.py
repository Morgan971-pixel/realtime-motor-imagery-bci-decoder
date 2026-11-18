"""The PhysioNet EEG Motor Movement/Imagery Database (EEGMMIDB) as club-layout recordings.

Source: Schalk et al., "BCI2000: a general-purpose brain-computer interface (BCI) system",
IEEE Trans. Biomed. Eng. 51(6), 2004; Goldberger et al., "PhysioBank, PhysioToolkit, and
PhysioNet", Circulation 101(23), 2000. Licence: Open Data Commons Attribution v1.0. 109 people,
64 channels at 160 Hz. In runs 4, 8 and 12 the person imagines opening and closing the left
fist (annotation T1) or the right fist (T2) while a target is on screen (about 4 s); T0 is rest.

Conversion: keep the 8 channels the club headset has (Fz, C3, Cz, C4, Pz, PO7, Oz, PO8, in the
club order), in microvolts; timestamps = sample index / 160; each T1/T2 onset becomes a one-
sample stim pulse with the club code trial * 100 + 30 + movement (phase 3 = imagery, movement
1 = left, 2 = right; trial counts the cues in the run from 1). T0 adds nothing.

Layout written by fetch: OUT/subjectNNN/sessionRRR/SNNNRrr.csv, one session per run, so
`python -m mi_ref.evaluate --data-dir OUT --fs 160 --min-trials 10` reads it unchanged. A run
whose file is refused gets SNNNRrr.refused (the reason) instead of a CSV.

Every download is checked against the SHA-256 pinned in eegmmidb_sha256.txt (taken from
PhysioNet's SHA256SUMS.txt for version 1.0.0); a mismatch stops the fetch.
"""
import argparse
import hashlib
import http.client
import re
import shutil
import sys
import urllib.request
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .edf import Edf, read_edf
from .epochs import to_samples
from .events import IMAGERY_PHASE, LEFT, RIGHT
from .recording import EEG_CHANNELS, Recording, save_recording

FS = 160.0
IMAGERY_RUNS = (4, 8, 12)
SUBJECTS = range(1, 110)
BASE_URL = "https://physionet.org/files/eegmmidb/1.0.0"
SHA256_FILE = Path(__file__).with_name("eegmmidb_sha256.txt")
_CUES = {"T1": LEFT, "T2": RIGHT}


def _norm(label: str) -> str:
    """EEGMMIDB pads labels with dots to 4 characters ('C3..', 'Po7.'); compare without them."""
    return label.rstrip(".").upper()


def to_recording(edf: Edf, run: int) -> Recording:
    if run not in IMAGERY_RUNS:
        raise ValueError(f"run {run} is not a left/right imagery run {IMAGERY_RUNS}")
    names = [_norm(label) for label in edf.labels]
    picks = []
    for ch in EEG_CHANNELS:
        idx = [i for i, name in enumerate(names) if name == ch.upper()]
        if len(idx) != 1:
            raise ValueError(f"channel {ch}: expected exactly one signal, found {len(idx)}")
        i = idx[0]
        if edf.fs[i] != FS:
            raise ValueError(f"channel {ch}: sample rate {edf.fs[i]:g} Hz, expected {FS:g}")
        if edf.units[i] != "uV":
            raise ValueError(f"channel {ch}: unit {edf.units[i]!r}, expected 'uV'")
        picks.append(i)
    signal = np.column_stack([edf.signals[i] for i in picks])
    n = len(signal)
    stim = np.zeros(n, dtype=np.int64)
    last, trial = -1, 0
    for a in edf.annotations:
        if a.text == "T0":
            continue
        if a.text not in _CUES:
            raise ValueError(f"unexpected annotation {a.text!r} at {a.onset} s")
        if a.onset < 0:
            raise ValueError(f"cue at negative time {a.onset} s")
        sample = to_samples(a.onset, FS)
        if sample >= n:
            raise ValueError(f"cue at {a.onset} s is past the end of the recording")
        if sample <= last:
            raise ValueError(f"cue at {a.onset} s is not after the previous cue")
        trial += 1
        stim[sample] = trial * 100 + IMAGERY_PHASE * 10 + _CUES[a.text]
        last = sample
    if trial == 0:
        raise ValueError("no T1/T2 cues")
    return Recording(np.arange(n) / FS, signal, stim)


def pinned_sha256(path: Path = SHA256_FILE) -> dict[str, str]:
    """{'S001/S001R04.edf': hex digest} from lines '<digest> <name>'."""
    out = {}
    for n, line in enumerate(path.read_text().splitlines(), 1):
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"{path.name} line {n}: expected '<digest> <name>', got {line!r}")
        digest, name = fields
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"{path.name} line {n}: {digest!r} is not a 64-digit hex digest")
        out[name] = digest
    return out


def parse_subjects(spec: str, subjects: range = SUBJECTS) -> list[int]:
    """'1-109', '1,5,7-9' -> sorted unique subject numbers, each within subjects (1..109 here)."""
    out: set[int] = set()
    for part in spec.split(","):
        lo, sep, hi = part.partition("-")
        if not lo.isdigit() or (sep and not hi.isdigit()):
            raise ValueError(f"bad subject range {part!r}")
        first, last = int(lo), int(hi) if sep else int(lo)
        if not (subjects.start <= first <= last < subjects.stop):
            raise ValueError(f"subjects must be within {subjects.start}-{subjects.stop - 1}, got {part!r}")
        out.update(range(first, last + 1))
    return sorted(out)


def download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "mi-realtime-fetch"})
    with urllib.request.urlopen(req, timeout=60) as resp, dest.open("wb") as fh:
        shutil.copyfileobj(resp, fh)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(out: Path, subjects: list[int], *, keep_edf: bool = False,
          get: Callable[[str, Path], None] | None = None, sums: dict[str, str] | None = None) -> dict[str, int]:
    """Download, check, convert and (unless keep_edf) delete each imagery run. Runs that already
    have a CSV or a .refused file are skipped, so an interrupted fetch can simply be rerun."""
    sums = pinned_sha256() if sums is None else sums
    get = download if get is None else get  # looked up per call, so tests can replace download
    counts = {"converted": 0, "present": 0, "refused": 0}
    edf_dir = out / "edf"
    edf_dir.mkdir(parents=True, exist_ok=True)
    for s in subjects:
        for run in IMAGERY_RUNS:
            name = f"S{s:03d}R{run:02d}"
            session = out / f"subject{s:03d}" / f"session{run:03d}"
            csv, refused = session / f"{name}.csv", session / f"{name}.refused"
            if csv.exists() or refused.exists():
                counts["present"] += 1
                continue
            key = f"S{s:03d}/{name}.edf"
            if key not in sums:
                raise ValueError(f"{key}: no pinned SHA-256")
            edf = edf_dir / f"{name}.edf"
            if not edf.exists():
                part = edf.with_suffix(".part")
                get(f"{BASE_URL}/{key}", part)
                part.replace(edf)
            if _sha256(edf) != sums[key]:
                edf.unlink()
                raise ValueError(f"{key}: SHA-256 does not match the pinned value (file deleted)")
            session.mkdir(parents=True, exist_ok=True)
            try:
                rec = to_recording(read_edf(edf.read_bytes()), run)
            except ValueError as err:
                refused.write_text(f"{err}\n")
                print(f"{name}: refused: {err}", file=sys.stderr)
                counts["refused"] += 1
            else:
                tmp = session / f"{name}.csv.part"
                save_recording(tmp, rec)
                tmp.replace(csv)
                counts["converted"] += 1
            if not keep_edf:
                edf.unlink()
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m mi_ref.eegmmidb", description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True, help="output folder (keep it outside git)")
    ap.add_argument("--subjects", default="1-109", help="e.g. 1-109 (default) or 1,5,7-9")
    ap.add_argument("--keep-edf", action="store_true", help="keep the downloaded EDF files in OUT/edf")
    a = ap.parse_args(argv)
    try:
        subjects = parse_subjects(a.subjects)
    except ValueError as err:
        ap.error(str(err))
    try:
        counts = fetch(a.out, subjects, keep_edf=a.keep_edf)
    except (OSError, ValueError, http.client.HTTPException) as err:  # IncompleteRead is not an OSError
        print(f"error: {err}", file=sys.stderr)
        return 1
    print(f"{counts['converted']} converted, {counts['present']} already present, {counts['refused']} refused")
    return 0


if __name__ == "__main__":
    sys.exit(main())
