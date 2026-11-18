"""Every replay command (Python, Rust `mi-replay`, C++ `mi_replay`) speaks the same interface.

The native commands are the release builds the benchmark uses; one that is not built is
skipped unless REQUIRE_CORES names its core (mi_core = Rust, mi_core_cpp = C++).
"""
import json
import os
import subprocess

import pytest

from helpers import REQUIRED
from mi_ref.bench import COMMANDS, stale
from mi_ref.cores import NATIVE
from mi_ref.golden import GOLDEN_DIR

EXPECTED = json.loads((GOLDEN_DIR / "expected.json").read_text())
MODEL, EDGE = str(GOLDEN_DIR / "model.json"), str(GOLDEN_DIR / "edge_stream.csv")
RAIL_MODEL, RAIL_STREAM = str(GOLDEN_DIR / "rail_model.json"), str(GOLDEN_DIR / "rail_stream.csv")


@pytest.fixture(params=sorted(COMMANDS))
def command(request):
    cmd = COMMANDS[request.param]
    if request.param in NATIVE and not os.path.exists(cmd[0]):
        if request.param in REQUIRED:
            pytest.fail(f"{cmd[0]} is not built")
        pytest.skip(f"{cmd[0]} is not built")
    if reason := stale(request.param):  # a stale binary would test yesterday's code
        pytest.fail(reason)
    return request.param, cmd


def _run(cmd, *args):
    return subprocess.run([*cmd, *args], capture_output=True, text=True)


def test_every_command_prints_the_same_summary(command):
    impl, cmd = command
    out = _run(cmd, MODEL, EDGE, "--chunk", "25", "--repeat", "2")
    assert out.returncode == 0, out.stderr
    s = json.loads(out.stdout)
    assert set(s) == {"impl", "chunk", "paced", "repeat", "samples", "elapsed_ns", "decisions", "latencies_ns"}
    assert (s["impl"], s["chunk"], s["paced"], s["repeat"], s["samples"]) == (impl, 25, False, 2, 2 * 9250)
    assert s["decisions"][:9] == [[d["sample"], d["label"]] for d in EXPECTED["edge_stream"]["decisions"]]
    # 20, not 2 x 9: the onsets at samples 8626 and 9150 lack a full window in the first pass,
    # and the second pass continues the stream, so their decisions come due at 9250 and 9774
    assert s["decisions"][9:11] == [[9250, 2], [9774, 1]]
    assert len(s["decisions"]) == len(s["latencies_ns"]) == 20
    assert all(isinstance(x, int) and 0 <= x <= s["elapsed_ns"] for x in s["latencies_ns"])


def test_every_command_refuses_where_the_rail_golden_does(command):
    # the summary lists [sample, label] per decision; label 0 is a refusal (pinned is not printed)
    out = _run(command[1], RAIL_MODEL, RAIL_STREAM)
    assert out.returncode == 0, out.stderr
    decisions = json.loads(out.stdout)["decisions"]
    assert decisions == [[d["sample"], d["label"]] for d in EXPECTED["rail_stream"]["decisions"]]
    assert [sample for sample, label in decisions if label == 0] == [1874, 4624, 12874, 15624, 21124]


def test_option_values_may_follow_an_equals_sign(command):
    spaced = json.loads(_run(command[1], MODEL, EDGE, "--chunk", "7", "--repeat", "2").stdout)
    joined = json.loads(_run(command[1], MODEL, EDGE, "--chunk=7", "--repeat=2").stdout)
    assert (joined["chunk"], joined["repeat"]) == (7, 2)
    assert joined["decisions"] == spaced["decisions"]


def test_paced_mode_runs_at_the_sampling_rate(command, tmp_path):
    short = tmp_path / "short.csv"  # header + 50 samples = two 25-sample chunks, due at 0.1 s and 0.2 s
    short.write_text("".join((GOLDEN_DIR / "edge_stream.csv").read_text().splitlines(keepends=True)[:51]))
    out = _run(command[1], MODEL, str(short), "--paced")
    assert out.returncode == 0, out.stderr
    s = json.loads(out.stdout)
    assert (s["paced"], s["chunk"], s["samples"]) == (True, 25, 50)
    assert s["elapsed_ns"] >= 200_000_000


def test_help_prints_usage_and_exits_0(command):
    out = _run(command[1], "-h")
    assert out.returncode == 0
    assert out.stdout.startswith("usage:")


BAD_ARGS = [
    [], [MODEL], [MODEL, EDGE, EDGE],                    # not exactly two positional arguments
    [MODEL, EDGE, "--chunk"],                            # missing value
    [MODEL, EDGE, "--chunk", "0"], [MODEL, EDGE, "--repeat", "x"], [MODEL, EDGE, "--chunk", "+5"],
    [MODEL, EDGE, "--chunk", "1_0"], [MODEL, EDGE, "--chunk", "18446744073709551616"],  # not a count
    [MODEL, EDGE, "--fast"], [MODEL, EDGE, "--rep", "2"], [MODEL, EDGE, "-x"],       # unknown options
    [MODEL, EDGE, "--paced=yes"],
    ["--", MODEL, EDGE], [MODEL, "--", EDGE], ["-1", EDGE],                         # argparse alone accepts these
    [MODEL, EDGE, "--bogus", "-h"],                      # the first bad argument wins over a later -h
]


@pytest.mark.parametrize("args", BAD_ARGS, ids=lambda a: " ".join(a)[-40:] or "none")
def test_bad_arguments_print_usage_and_exit_2(command, args):
    out = _run(command[1], *args)
    assert out.returncode == 2
    assert "usage:" in out.stderr
    assert out.stdout == ""


def test_a_repeat_count_that_overflows_exits_2(command):
    if command[0] == "python":
        pytest.skip("Python integers do not overflow; it would just run for a very long time")
    out = _run(command[1], MODEL, EDGE, "--repeat", "18446744073709551615")
    assert out.returncode == 2
    assert "usage:" in out.stderr


@pytest.mark.parametrize("which", ["model", "csv", "csv-directory", "csv-header-only"])
def test_a_bad_file_is_reported_with_its_path(command, tmp_path, which):
    bad = {"model": tmp_path / "missing.json", "csv": tmp_path / "missing.csv",
           "csv-directory": tmp_path, "csv-header-only": tmp_path / "empty.csv"}[which]
    if which == "csv-header-only":
        bad.write_text("timestamps,Fz,C3,Cz,C4,Pz,PO7,Oz,PO8,stim\n")
    out = _run(command[1], *((str(bad), EDGE) if which == "model" else (MODEL, str(bad))))
    assert out.returncode == 1
    assert out.stderr.startswith(f"{bad}: ")
    assert not out.stderr[len(f"{bad}: "):].startswith(f"{bad}: "), out.stderr  # no "PATH: PATH: reason"
    assert out.stdout == ""
