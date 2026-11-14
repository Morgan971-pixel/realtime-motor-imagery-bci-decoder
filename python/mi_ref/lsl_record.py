"""Record a cued calibration session from LSL into a club-layout CSV, outside the repository.

The EEG channels are picked as in mi_ref.lsl_live (--channels), the cue markers are merged in
exactly as the live decoder merges them (--cue-stream), and the result is written as
timestamps,Fz,...,PO8,stim with the LSL timestamps, so mi_ref.train and mi_ref.evaluate read it
like any club recording. Record only with the club's OK and the wearer's consent.
"""
import argparse
import sys

import numpy as np

from .cue_merge import CueMerger
from .events import validate_stim
from .lsl_live import _drain, _pylsl, open_streams, parse_channels, samples
from .paths import outside_repo
from .recording import Recording, save_recording


def record(inlet, *, fs, eeg_indices=None, stim_index=None, cue_inlet=None, cue_lag=0.5, max_samples=None,
           timeout=0.25, pylsl_module=None, report=None) -> Recording:
    """Collect samples until max_samples, a lost stream or Ctrl+C; returns them as a Recording."""
    pylsl_module = pylsl_module or _pylsl()
    report = report or (lambda message: print(message, file=sys.stderr, flush=True))
    if cue_inlet is not None and stim_index is not None:
        raise ValueError("the EEG stream has its own stim channel; use it or a cue stream, not both")
    merger = CueMerger(fs, cue_lag, report) if cue_inlet is not None else None
    lost = getattr(getattr(pylsl_module, "util", None), "LostError", None)
    lost_errors = (lost,) if lost else ()
    rows, stims, stamps = [], [], []
    inlet.open_stream(timeout=timeout)
    if cue_inlet is not None:
        cue_inlet.open_stream(timeout=timeout)
    try:
        for eeg, stim, stamp in samples(inlet, eeg_indices=eeg_indices, stim_index=stim_index, cue_inlet=cue_inlet,
                                        merger=merger, timeout=timeout, max_samples=max_samples,
                                        lost=lost_errors, report=report):
            rows.append(eeg)
            stims.append(stim)
            stamps.append(stamp)
    except KeyboardInterrupt:
        report("stopped early; saving what was recorded")
        if merger is not None:
            try:
                _drain(cue_inlet, merger, 0.0)  # cues already waiting for the held samples
            except lost_errors as err:
                report(f"LSL cue stream lost: {err}")
            for eeg, stim, stamp in merger.flush():
                rows.append(eeg)
                stims.append(stim)
                stamps.append(stamp)
    finally:
        inlet.close_stream()
        if cue_inlet is not None:
            cue_inlet.close_stream()
    if not rows:
        raise ValueError("no samples were recorded")
    if merger is not None and merger.summary():
        report(merger.summary())
    return Recording(np.array(stamps), np.vstack(rows), validate_stim(np.array(stims, dtype=np.float64)))


def write_new(path, rec: Recording) -> None:
    """Save rec as a club-layout CSV; FileExistsError instead of overwriting anything."""
    with open(path, "x", encoding="utf-8", newline="") as fh:
        save_recording(fh, rec)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m mi_ref.lsl_record", description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="new CSV file, outside the repository")
    ap.add_argument("--seconds", type=float, required=True, help="how long to record")
    ap.add_argument("--fs", type=float, default=250.0, help="the EEG stream's sample rate (default 250, the Unicorn)")
    ap.add_argument("--stream-name", default="Unicorn")
    ap.add_argument("--stream-type", default="EEG")
    ap.add_argument("--channels", help="as in mi_ref.lsl_live, e.g. 0-7")
    ap.add_argument("--cue-stream", default="mi-realtime-cues", help="cue marker stream (default mi-realtime-cues)")
    ap.add_argument("--cue-lag", type=float, default=0.5)
    ap.add_argument("--timeout", type=float, default=10.0)
    a = ap.parse_args(argv)
    try:
        out = outside_repo(a.out)
    except ValueError as err:
        ap.error(str(err))
    if out.exists():
        ap.error(f"{a.out} already exists; choose a new file")
    if not (np.isfinite(a.seconds) and a.seconds > 0) or not (np.isfinite(a.fs) and a.fs > 0):
        ap.error("--seconds and --fs must be finite and > 0")
    try:
        pylsl = _pylsl()
        channels = parse_channels(a.channels) if a.channels else None
        inlet, eeg_indices, stim_index, cue_inlet = open_streams(
            pylsl, a.fs, a.stream_name, a.stream_type, channels, a.cue_stream, a.timeout)
        print(f"recording {a.seconds:g} s; Ctrl+C stops early and still saves", file=sys.stderr)
        rec = record(inlet, fs=a.fs, eeg_indices=eeg_indices, stim_index=stim_index, cue_inlet=cue_inlet,
                     cue_lag=a.cue_lag, max_samples=round(a.seconds * a.fs), pylsl_module=pylsl)
        out.parent.mkdir(parents=True, exist_ok=True)
        write_new(out, rec)
    except KeyboardInterrupt:
        print("stopped before recording started", file=sys.stderr)
        return 1
    except (OSError, ValueError, RuntimeError, TimeoutError) as err:
        print(str(err), file=sys.stderr)
        return 1
    print(f"wrote {len(rec.stim)} samples to {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
