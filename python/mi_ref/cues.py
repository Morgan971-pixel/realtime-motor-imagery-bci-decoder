"""Show left/right motor-imagery cues and publish each one as an LSL marker.

Every trial has three phases, each announced on screen and sent as a marker with the club's
code trial * 100 + phase * 10 + movement: phase 2 = get ready, 3 = imagine (the decoder's
window starts 0.5 s after this marker and lasts 2 s), 5 = rest; movement 1 = left, 2 = right.
Left and right trials are balanced and shuffled with --seed.

The marker stream (default name mi-realtime-cues, type Markers) has one int32 channel and an
irregular rate. Each marker is stamped with the LSL clock at the moment its cue is shown.
"""
import argparse
import math
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .epochs import WINDOW_LEN_S, WINDOW_START_S
from .events import IMAGERY_PHASE, LEFT, RIGHT

PREPARE_PHASE, REST_PHASE = 2, 5
MIN_IMAGERY_S = WINDOW_START_S + WINDOW_LEN_S  # the decision window must end inside the cue
NAMES = {LEFT: "LEFT", RIGHT: "RIGHT"}


@dataclass(frozen=True)
class Cue:
    t: float  # seconds after the start
    code: int
    text: str


def schedule(n_trials: int = 40, seed: int = 0, prepare_s: float = 3.0, imagery_s: float = 5.0,
             rest_s: float = 3.0, lead_in_s: float = 3.0) -> list[Cue]:
    if n_trials < 2 or n_trials % 2:
        raise ValueError(f"trials must be an even number >= 2 (half left, half right), got {n_trials}")
    for name, value in (("prepare", prepare_s), ("rest", rest_s)):
        if not (math.isfinite(value) and value > 0):
            raise ValueError(f"{name} time must be finite and > 0, got {value}")
    if not (math.isfinite(imagery_s) and imagery_s >= MIN_IMAGERY_S):
        raise ValueError(f"imagery time must be at least {MIN_IMAGERY_S:g} s (the decision window ends "
                         f"{MIN_IMAGERY_S:g} s after the cue), got {imagery_s}")
    if not (math.isfinite(lead_in_s) and lead_in_s >= 0):
        raise ValueError(f"lead-in must be finite and >= 0, got {lead_in_s}")
    moves = np.random.default_rng(seed).permutation([LEFT] * (n_trials // 2) + [RIGHT] * (n_trials // 2))
    cues, t = [], lead_in_s
    for trial, move in enumerate(moves.tolist(), 1):
        name = NAMES[move]
        for phase, text, duration in ((PREPARE_PHASE, f"get ready: {name}", prepare_s),
                                      (IMAGERY_PHASE, f"IMAGINE {name} HAND", imagery_s),
                                      (REST_PHASE, "rest", rest_s)):
            cues.append(Cue(t, trial * 100 + phase * 10 + move, text))
            t += duration
    return cues


def show_terminal(text: str) -> None:
    if text.startswith("IMAGINE"):
        print(f"\n{'=' * 32}\n   {text}\n{'=' * 32}", flush=True)
    else:
        print(f"\n   {text}", flush=True)


def marker_info(name: str, source_id: str, pylsl):
    return pylsl.StreamInfo(name, "Markers", 1, pylsl.IRREGULAR_RATE, pylsl.cf_int32, source_id)


def present(cues: list[Cue], outlet, local_clock: Callable[[], float],
            show: Callable[[str], None] = show_terminal, *, clock=time.monotonic, sleep=time.sleep) -> int:
    """Show each cue at its time (absolute deadlines, so delays do not accumulate) and push its
    code stamped with the LSL clock at that moment."""
    start = clock()
    for cue in cues:
        delay = start + cue.t - clock()
        if delay > 0:
            sleep(delay)
        show(cue.text)
        outlet.push_sample([cue.code], timestamp=local_clock())
    return len(cues)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m mi_ref.cues", description=__doc__.splitlines()[0])
    ap.add_argument("--trials", type=int, default=40, help="even number of trials (default 40)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prepare-s", type=float, default=3.0)
    ap.add_argument("--imagery-s", type=float, default=5.0)
    ap.add_argument("--rest-s", type=float, default=3.0)
    ap.add_argument("--lead-in-s", type=float, default=3.0, help="wait before the first cue (default 3)")
    ap.add_argument("--name", default="mi-realtime-cues", help="LSL marker stream name")
    a = ap.parse_args(argv)
    try:
        cues = schedule(a.trials, a.seed, a.prepare_s, a.imagery_s, a.rest_s, a.lead_in_s)
    except ValueError as err:
        ap.error(str(err))
    try:
        import pylsl
    except ImportError:
        print("LSL support is optional; install with .venv/bin/pip install -e 'python[live]'", file=sys.stderr)
        return 1
    outlet = pylsl.StreamOutlet(marker_info(a.name, f"mi-realtime-cues-{uuid.uuid4()}", pylsl))
    total = cues[-1].t + a.rest_s
    print(f"{a.trials} trials, {total:.0f} s; markers on LSL stream {a.name!r}. Ctrl+C stops.", file=sys.stderr)
    try:
        present(cues, outlet, pylsl.local_clock)
    except KeyboardInterrupt:
        return 0
    show_terminal("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
