"""Train a model from calibration recordings and report its honest cross-validated accuracy.

The accuracy comes from mi_ref.evaluate's contiguous-block cross-validation, next to the 95%
chance threshold for that many trials. Below the threshold, live decisions are close to guesses:
record more trials or check the headset before trusting them. The model file is written outside
the repository; trained models never enter git.
"""
import argparse
import sys

import numpy as np

from .evaluate import cross_validate
from .model import train
from .paths import outside_repo
from .recording import load_recording


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m mi_ref.train", description=__doc__.splitlines()[0])
    ap.add_argument("csv", nargs="+", help="calibration recordings (club CSV layout)")
    ap.add_argument("--out", required=True, help="new model JSON file, outside the repository")
    ap.add_argument("--fs", type=float, default=250.0, help="sample rate of the recordings (default 250)")
    ap.add_argument("--rail-abs", type=float, help="refuse decisions when a used channel sits at this raw value")
    a = ap.parse_args(argv)
    try:
        out = outside_repo(a.out)
    except ValueError as err:
        ap.error(str(err))
    if out.exists():
        ap.error(f"{a.out} already exists; choose a new file")
    if not (np.isfinite(a.fs) and a.fs > 0):
        ap.error("--fs must be finite and > 0")
    try:
        recordings = [load_recording(path) for path in a.csv]
        cv = cross_validate(recordings, fs=a.fs)
        model = train(recordings, fs=a.fs, rail_abs=a.rail_abs)
    except KeyboardInterrupt:
        print("stopped; no model written", file=sys.stderr)
        return 1
    except (OSError, ValueError) as err:
        print(str(err), file=sys.stderr)
        return 1
    verdict = ("above chance" if cv["accuracy"] >= cv["chance_95"]
               else "NOT above chance: live decisions will be close to guesses")
    print(f"cross-validated accuracy {cv['accuracy']:.1%} on {cv['n_trials']} trials, "
          f"chance_95 {cv['chance_95']:.1%}: {verdict}")
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "x", encoding="utf-8") as fh:  # never overwrite, even a file made meanwhile
            fh.write(model.to_json())
    except OSError as err:
        print(str(err), file=sys.stderr)
        return 1
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
