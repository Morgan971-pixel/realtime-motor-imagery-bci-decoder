"""Honest per-subject accuracy: fixed settings, whole (recording, trial number) groups held out
in contiguous blocks of time, and a binomial chance threshold.

Reading the numbers:
- chance_95 treats trials as independent. Trials inside a calibration block share a hand and
  are correlated, so the threshold is somewhat lenient: treat a narrow pass with suspicion.
- With N subjects, about 0.05 * N pass by luck alone even when nothing is decodable.

--exclude-pinned RAIL drops, per subject, every channel whose raw value is at or beyond RAIL
in at least half of that subject's samples (an electrode stuck at the amplifier's limit carries
no EEG). The choice never looks at labels, so it cannot leak test trials into training.
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binom

from .events import imagery_markers, markers_from_stim
from .filters import DEFAULT_BANDS, design_band
from .lda import predict
from .model import EpochSet, build_epochs, features, fit_epochs
from .recording import EEG_CHANNELS, FS, Recording, load_recording

N_PAIRS = 2  # CSP pairs per band, as train's default
MIN_CHANNELS = 2 * N_PAIRS  # CSP with n pairs needs at least 2n channels
PINNED_FRACTION = 0.5


def chance_threshold(n: int, alpha: float = 0.05) -> float:
    """Smallest accuracy a coin-flipper reaches with probability <= alpha on n trials."""
    k = int(binom.ppf(1.0 - alpha, n, 0.5)) + 1
    return k / n


def contiguous_folds(groups: np.ndarray, n_folds: int) -> list[np.ndarray]:
    """Split the time-ordered groups into contiguous blocks; returns test indices per fold.

    Contiguous blocks keep held-out trials away from their training neighbours in time, the
    conservative choice against slow drift.
    """
    uniq = np.unique(groups)
    blocks = np.array_split(uniq, min(n_folds, len(uniq)))
    return [np.flatnonzero(np.isin(groups, block)) for block in blocks]


def _subset(es: EpochSet, idx: np.ndarray) -> EpochSet:
    return EpochSet([Xb[idx] for Xb in es.X], es.y[idx], es.groups[idx])


def usable_channels(recordings: list[Recording], rail_abs: float) -> np.ndarray:
    """Channels at or beyond rail_abs in fewer than half of the samples, pooled over recordings."""
    pinned = np.mean(np.vstack([np.abs(r.signal) >= rail_abs for r in recordings]), axis=0)
    return np.flatnonzero(pinned < PINNED_FRACTION)


def cross_validate(recordings: list[Recording], n_folds: int = 5, fs: float = FS, keep=None) -> dict:
    sos_bank = [design_band(lo, hi, fs) for lo, hi in DEFAULT_BANDS]
    es = build_epochs(recordings, sos_bank, fs)
    n_groups = len(np.unique(es.groups))
    if n_groups < 2:
        raise ValueError(f"need at least two trial groups to cross-validate, got {n_groups}")
    correct = 0
    for test in contiguous_folds(es.groups, n_folds):
        train_idx = np.setdiff1d(np.arange(len(es.y)), test)
        csp, w, b = fit_epochs(_subset(es, train_idx), n_pairs=N_PAIRS, keep=keep)
        correct += int(np.sum(predict(w, b, features(csp, _subset(es, test).X)) == es.y[test]))
    n = len(es.y)
    return {"n_trials": n, "n_groups": n_groups, "accuracy": correct / n, "chance_95": chance_threshold(n)}


def find_subjects(data_dir: Path, min_trials: int = 30) -> tuple[dict[str, list[Recording]], list[Path]]:
    """Recordings with >= min_trials imagery markers, grouped by subject folder (each loaded once),
    plus the paths that were skipped."""
    subjects: dict[str, list[Recording]] = defaultdict(list)
    skipped: list[Path] = []
    for csv in sorted(data_dir.glob("subject*/session*/*.csv")):
        rec = load_recording(csv)
        if len(imagery_markers(markers_from_stim(rec.stim))) >= min_trials:
            subjects[csv.parent.parent.name].append(rec)
        else:
            skipped.append(csv)
    return dict(subjects), skipped


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--exclude-pinned", type=float, metavar="RAIL",
                    help="drop channels at or beyond RAIL in at least half of a subject's samples")
    ap.add_argument("--fs", type=float, default=FS, help=f"sample rate of the recordings in Hz (default {FS:g})")
    ap.add_argument("--min-trials", type=int, default=30, metavar="N",
                    help="skip a recording with fewer than N imagery trials (default 30)")
    args = ap.parse_args(argv)
    if not (np.isfinite(args.fs) and args.fs > 0):
        ap.error("--fs needs a finite rate > 0")
    if args.min_trials < 1:
        ap.error("--min-trials needs N >= 1")
    if args.exclude_pinned is not None and not (np.isfinite(args.exclude_pinned) and args.exclude_pinned > 0):
        ap.error("--exclude-pinned needs a finite RAIL > 0")
    subjects, skipped = find_subjects(args.data_dir, args.min_trials)
    print(f"{len(skipped)} recording(s) skipped for having fewer than {args.min_trials} imagery trials")
    if not subjects:
        print("no subjects with enough trials")
        return 1
    print(f"{'subject':12} {'trials':>6} {'groups':>6} {'accuracy':>9} {'chance95':>9}  above?")
    accs, above = [], 0
    for subject, recs in subjects.items():
        keep, note = None, ""
        if args.exclude_pinned is not None:
            keep = usable_channels(recs, args.exclude_pinned)
            if len(keep) < MIN_CHANNELS:
                print(f"{subject:12} skipped: only {len(keep)} usable channels")
                continue
            dropped = [EEG_CHANNELS[c] for c in range(len(EEG_CHANNELS)) if c not in keep]
            note = f"  dropped {', '.join(dropped)}" if dropped else ""
        try:
            r = cross_validate(recs, fs=args.fs, keep=keep)
        except ValueError as err:
            print(f"{subject:12} skipped: {err}")
            continue
        accs.append(r["accuracy"])
        above += r["accuracy"] >= r["chance_95"]
        flag = "yes" if r["accuracy"] >= r["chance_95"] else "no"
        print(f"{subject:12} {r['n_trials']:6d} {r['n_groups']:6d} {r['accuracy']:9.1%} {r['chance_95']:9.1%}  {flag}{note}")
    if accs:
        print(f"{'mean':12} {'':6} {'':6} {np.mean(accs):9.1%}")
        print(f"{above} of {len(accs)} subjects above chance_95; about {0.05 * len(accs):.2f} expected by luck alone")
    return 0


if __name__ == "__main__":
    sys.exit(main())
