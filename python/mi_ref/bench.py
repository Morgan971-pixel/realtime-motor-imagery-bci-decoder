"""Benchmark the three replay commands: decision latency, throughput and peak memory.

    python -m mi_ref.bench                         # prints a Markdown table
    python -m mi_ref.bench --readme ../README.md   # also rewrites the table in the README

Build the native commands first (release builds; see the README's Benchmarks section); a
release binary older than its sources is refused. Every command runs as its own process, so
peak memory is that process's maximum resident set size (for Python this includes the
interpreter and NumPy). Before printing, the script checks that all implementations made
exactly the same decisions in each mode.
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .golden import GOLDEN_DIR
from .model import Model

REPO = GOLDEN_DIR.parent
COMMANDS = {
    "python": [sys.executable, "-m", "mi_ref.replay"],
    "rust": [str(REPO / "rust" / "mi_core" / "target" / "release" / "mi-replay")],
    "cpp": [str(REPO / "cpp" / "build" / "release" / "mi_replay")],
}
# what each release binary is built from: a newer file here means the binary is stale
SOURCES = {
    "rust": [REPO / "rust" / "mi_core" / "src", REPO / "rust" / "mi_core" / "Cargo.toml"],
    "cpp": [REPO / "cpp" / "src", REPO / "cpp" / "include", REPO / "cpp" / "tools", REPO / "cpp" / "CMakeLists.txt"],
}
# the commands that rebuild each release binary, run from the repository root
BUILD = {
    "rust": "(cd rust/mi_core && cargo build --release --bin mi-replay)",
    "cpp": "(cd cpp && cmake --preset release && cmake --build --preset release)",
}
LABELS = {"python": "Python (reference)", "rust": "Rust", "cpp": "C++"}
MIN_FOR_P99 = 100  # below this many decisions, p99 is just the maximum: print n/a
MODES = {"fast": ["--chunk", "1"], "paced": ["--chunk", "25", "--paced"]}
START, END = "<!-- bench:start -->", "<!-- bench:end -->"


@dataclass
class Run:
    impl: str
    mode: str
    summary: dict
    peak_rss_bytes: int


def stale(impl: str) -> str | None:
    """Why `impl`'s release binary cannot be trusted (older than a source file), or None."""
    if impl not in SOURCES or not os.path.exists(COMMANDS[impl][0]):
        return None
    built = os.path.getmtime(COMMANDS[impl][0])
    files = [f for root in SOURCES[impl] for f in ([root] if root.is_file() else sorted(root.rglob("*"))) if f.is_file()]
    newer = [f for f in files if f.stat().st_mtime > built]
    if not newer:
        return None
    return f"{COMMANDS[impl][0]} is older than {newer[0].relative_to(REPO)}: rebuild it with {BUILD[impl]}"


def nearest_rank(values: list[int], p: int) -> int:
    """The smallest value with at least p% of the values at or below it (p a whole percent)."""
    if not values:
        raise ValueError("no values")
    s = sorted(values)
    return s[max(1, -(-p * len(s) // 100)) - 1]  # integer ceiling: no float rounding


def run_command(impl: str, mode: str, cmd: list[str], expect: dict | None = None) -> Run:
    """Run one replay command; os.wait4 reports the peak memory of that process alone.

    `expect` holds summary fields the command must echo back (for example the options it was
    given), so a stale or misbehaving binary cannot be reported under the wrong mode.
    """
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err)
        out = proc.stdout.read()
        proc.stdout.close()
        _, status, usage = os.wait4(proc.pid, 0)
        proc.returncode = os.waitstatus_to_exitcode(status)
        if proc.returncode != 0:
            err.seek(0)
            raise SystemExit(f"{impl} ({mode}) failed with exit {proc.returncode}: {err.read().decode().strip()}")
    # ru_maxrss is in bytes on macOS and in KiB on Linux
    peak = usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    s = json.loads(out)
    wrong = {k: s.get(k) for k, v in (expect or {}).items() if s.get(k) != v}
    if wrong:
        raise SystemExit(f"{impl} ({mode}) echoed {wrong}, expected {expect}")
    if not s["decisions"] or len(s["latencies_ns"]) != len(s["decisions"]) or s["elapsed_ns"] <= 0:
        raise SystemExit(f"{impl} ({mode}) made no decisions or reported inconsistent timings: nothing to report")
    return Run(impl, mode, s, peak)


def check_agreement(runs: list[Run]) -> None:
    for r in runs[1:]:
        for key in ("decisions", "samples"):
            if r.summary[key] != runs[0].summary[key]:
                raise SystemExit(f"{r.impl} disagrees with {runs[0].impl} on {key} in {r.mode} mode: refusing to report timings")


def table(runs: list[Run]) -> str:
    def us(ns: int) -> str:
        return f"{ns / 1000:.1f}"

    lines = ["| Implementation | Mode | Decisions | p50 (µs) | p99 (µs) | max (µs) | Samples/s | Peak RSS (MiB) |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in runs:
        lat, s = r.summary["latencies_ns"], r.summary
        rate = "n/a" if s["paced"] else f"{s['samples'] / (s['elapsed_ns'] / 1e9):,.0f}"
        p99 = us(nearest_rank(lat, 99)) if len(lat) >= MIN_FOR_P99 else "n/a"
        lines.append(f"| {LABELS[r.impl]} | {r.mode} | {len(lat)} | {us(nearest_rank(lat, 50))} | "
                     f"{p99} | {us(max(lat))} | {rate} | {r.peak_rss_bytes / 2**20:.1f} |")
    return "\n".join(lines)


def machine() -> str:
    chip = platform.processor() or platform.machine()
    if sys.platform == "darwin":
        try:
            chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return f"{chip}, {platform.system()} {platform.release()}"


def report(runs: list[Run], csv: Path, repeat: int, fs: float) -> str:
    """The caption describes only the modes that ran, at the model's sample rate."""
    samples = next((r.summary["samples"] // r.summary["repeat"] for r in runs), 0)
    lines = [f"Replay of `{csv.resolve().name}` ({samples:,} samples at {fs:g} Hz) on {machine()}, from `python -m mi_ref.bench`."]
    if any(r.mode == "fast" for r in runs):
        lines.append(f"**fast**: one sample at a time, as fast as possible, the recording played {repeat} times; "
                     "latency is the time `push` takes on a sample that completes a decision. "
                     "Samples/s counts the whole loop, which also reads the clock once per sample.")
    paced = next((r for r in runs if r.mode == "paced"), None)
    if paced:
        chunk, decisions = paced.summary["chunk"], paced.summary["decisions"]
        at_end = sum((sample + 1) % chunk == 0 for sample, _ in decisions)
        lines.append(f"**paced**: {chunk}-sample chunks delivered at {fs:g} Hz, played once, so the decoder idles between "
                     "chunks; latency runs from a chunk's handover to its decision (see docs/replay.md). "
                     f"{at_end} of {len(decisions)} decisions fall on the last sample of their chunk, so their latency "
                     "includes pushing the whole chunk.")
    if any(len(r.summary["latencies_ns"]) < MIN_FOR_P99 for r in runs):
        lines.append(f"p99 reads n/a below {MIN_FOR_P99} decisions, where it would only repeat the maximum.")
    return "\n".join([*lines, "", table(runs)])


def check_readme(path: Path) -> str:
    doc = path.read_text(encoding="utf-8")
    if doc.count(START) != 1 or doc.count(END) != 1 or doc.index(START) > doc.index(END):
        raise SystemExit(f"{path}: needs one {START} line followed by one {END} line")
    return doc


def update_readme(path: Path, text: str) -> None:
    head, rest = check_readme(path).split(START)
    _, tail = rest.split(END)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(f"{head}{START}\n{text}\n{END}{tail}", encoding="utf-8")
    os.replace(tmp, path)  # all or nothing: an interrupted write never leaves a half README


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m mi_ref.bench", description=__doc__.splitlines()[0])
    p.add_argument("--model", type=Path, default=GOLDEN_DIR / "model.json")
    p.add_argument("--csv", type=Path, default=GOLDEN_DIR / "stream.csv")
    p.add_argument("--impls", default="python,rust,cpp", help="comma-separated, from python,rust,cpp")
    p.add_argument("--modes", default="fast,paced", help="comma-separated, from fast,paced")
    p.add_argument("--repeat", type=int, default=20, help="passes over the recording in fast mode (default 20)")
    p.add_argument("--readme", type=Path, help="rewrite the table between the bench markers in this file")
    a = p.parse_args(argv)
    impls, modes = a.impls.split(","), a.modes.split(",")
    if not set(impls) <= set(COMMANDS) or not set(modes) <= set(MODES) or a.repeat < 1:
        p.error("unknown implementation or mode, or --repeat below 1")
    if a.readme and a.csv.resolve().parent != GOLDEN_DIR.resolve():
        p.error("--readme takes a golden/ recording only: real recordings and their names stay out of the README")
    if a.readme:
        check_readme(a.readme)  # before the runs, not after six minutes of them
    for impl in impls:
        if impl != "python" and not os.path.exists(COMMANDS[impl][0]):
            p.error(f"{COMMANDS[impl][0]} is not built: build it with {BUILD[impl]}")
        if reason := stale(impl):
            p.error(reason)
    fs = Model.from_json(a.model.read_text()).fs
    runs = []
    for mode in modes:
        repeat = a.repeat if mode == "fast" else 1
        mode_runs = []
        for impl in impls:
            print(f"running {impl} ({mode})...", file=sys.stderr)
            cmd = [*COMMANDS[impl], str(a.model), str(a.csv), *MODES[mode], "--repeat", str(repeat)]
            expect = {"impl": impl, "chunk": int(MODES[mode][1]), "paced": mode == "paced", "repeat": repeat}
            mode_runs.append(run_command(impl, mode, cmd, expect))
        check_agreement(mode_runs)
        runs += mode_runs
    text = report(runs, a.csv, a.repeat, fs)
    print(text)
    if a.readme:
        update_readme(a.readme, text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
