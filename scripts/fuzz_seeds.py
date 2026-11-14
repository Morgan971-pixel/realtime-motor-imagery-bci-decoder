"""Build the fuzz seed corpora from the synthetic golden files (never from recordings).

    python3 scripts/fuzz_seeds.py fuzz-corpus

writes fuzz-corpus/model (both golden models and every invalid and valid model in
golden/invalid), fuzz-corpus/csv (the first rows of each golden CSV and of every stream in the
manifest, written as CSV) and fuzz-corpus/stream (inputs for the Rust decode_stream target, in
its 4-bytes-per-row format). The output is git-ignored: rebuild it, do not commit it. Standard
library only.
"""
import json
import math
import shutil
import struct
import sys
from pathlib import Path

GOLDEN = Path(__file__).resolve().parent.parent / "golden"
HEADER = "timestamps,Fz,C3,Cz,C4,Pz,PO7,Oz,PO8,stim"
CSV_ROWS = 40  # header plus 39 samples: enough to reach every parser path, small enough to mutate fast
AT, DECIDE = 5, 629  # the seeds' imagery marker row, and the row both golden models decide at
C3 = 1 << 1  # rail-mask bit for C3, a used channel of golden/rail_model.json
L = 500  # both golden models' window length in samples


def stream_row(base: int, ctl: int, mask: int = 0) -> bytes:
    """One decode_stream row: a little-endian i16 base amplitude, the control byte, the rail mask."""
    return struct.pack("<hBB", base, ctl, mask)


def stream_seed(marker_ctl: int, mask=lambda k: 0, n: int = 700) -> bytes:
    """A 10 Hz wave with one marker at row AT; with either golden model's window (start 125
    samples, length 500) it decides at row AT + 125 + 499 = DECIDE, inside the seed."""
    return b"".join(stream_row(round(800 * math.sin(2 * math.pi * 10 * k / 250)), marker_ctl if k == AT else 0, mask(k))
                    for k in range(n))


def with_row(seed: bytes, before: int, row: bytes) -> bytes:
    """seed with one more row inserted before row BEFORE (4 bytes per row)."""
    return seed[:4 * before] + row + seed[4 * before:]


def csv_text(rows: list[list]) -> str:
    """A manifest stream (rows of 8 EEG values and stim) as a recording CSV, cut to CSV_ROWS lines."""
    body = [f"{i * 0.004:.3f}," + ",".join(str(v) for v in row) for i, row in enumerate(rows)]
    return "\n".join([HEADER, *body][:CSV_ROWS]) + "\n"


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__.strip().splitlines()[2].strip(), file=sys.stderr)
        return 2
    out = Path(argv[0])
    shutil.rmtree(out, ignore_errors=True)
    for sub in ("model", "csv", "stream"):
        (out / sub).mkdir(parents=True)

    for name in ("model.json", "rail_model.json"):
        shutil.copy(GOLDEN / name, out / "model" / f"golden_{name}")
    for f in sorted((GOLDEN / "invalid").glob("*.json")):
        if f.name != "manifest.json":
            shutil.copy(f, out / "model" / f.name)

    for f in sorted(GOLDEN.glob("*.csv")):
        lines = f.read_text().splitlines()[:CSV_ROWS]
        (out / "csv" / f.name).write_text("\n".join(lines) + "\n")
    manifest = json.loads((GOLDEN / "invalid" / "manifest.json").read_text())
    for s in manifest["streams"] + manifest["valid_streams"]:
        (out / "csv" / f"{s['name']}.csv").write_text(csv_text(s["rows"]))
    for s in manifest["rail_streams"]:  # rows are run-length encoded as [count, row] pairs
        (out / "csv" / f"{s['name']}.csv").write_text(csv_text([row for count, row in s["rows"] for _ in range(count)]))

    (out / "stream" / "imagery_left").write_bytes(stream_seed(10))
    (out / "stream" / "imagery_right").write_bytes(stream_seed(11))
    # invalid stim (50), NaN stim, and an infinite EEG value on channel 3, between valid rows
    (out / "stream" / "bad_rows").write_bytes(
        b"".join(stream_row(100 * k, ctl) for k, ctl in enumerate([0, 13, 0, 14, 0, 0x80 | 0x30, 10, 0])))
    # C3 at the rail on every row: the rail model refuses at DECIDE with pinned [1]
    (out / "stream" / "rail_pinned").write_bytes(stream_seed(10, lambda k: C3))
    # the same with one row in the middle of the window off the rail: the run resets, so it decides
    (out / "stream" / "rail_dip").write_bytes(stream_seed(10, lambda k: 0 if k == DECIDE - 250 else C3))
    # rejected rows inside a run at the rail must change nothing (as manifest rail_streams):
    # L - 1 accepted rows at the rail plus a rejected one (stim 50) at the rail still decides,
    last_l_minus_1 = stream_seed(10, lambda k: C3 if k > DECIDE - L + 1 else 0)
    (out / "stream" / "rail_rejected_on_rail").write_bytes(with_row(last_l_minus_1, DECIDE - 250, stream_row(0, 13, C3)))
    # and a whole window at the rail with a rejected row below it still refuses
    (out / "stream" / "rail_rejected_off_rail").write_bytes(
        with_row(stream_seed(10, lambda k: C3), DECIDE - 250, stream_row(0, 13, 0)))

    counts = {sub: len(list((out / sub).iterdir())) for sub in ("model", "csv", "stream")}
    print(f"seeds in {out}: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
