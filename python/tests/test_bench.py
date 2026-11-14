import os
import sys

import pytest

from mi_ref import bench
from mi_ref.golden import GOLDEN_DIR


def test_nearest_rank_percentiles():
    xs = [5, 1, 4, 2, 3]
    assert [bench.nearest_rank(xs, p) for p in (1, 20, 50, 99, 100)] == [1, 1, 3, 5, 5]
    assert bench.nearest_rank(list(range(1, 201)), 99) == 198
    assert bench.nearest_rank(list(range(1, 101)), 7) == 7  # float ceil(7 / 100 * 100) would give 8
    with pytest.raises(ValueError):
        bench.nearest_rank([], 50)


def _summary(impl, decisions, latencies, samples=1000, elapsed=2_000_000, paced=False):
    return {"impl": impl, "chunk": 1, "paced": paced, "repeat": 1, "samples": samples, "elapsed_ns": elapsed,
            "decisions": decisions, "latencies_ns": latencies}


def test_runs_that_disagree_are_refused():
    a = bench.Run("python", "fast", _summary("python", [[10, 1]], [100]), 0)
    b = bench.Run("rust", "fast", _summary("rust", [[10, 2]], [100]), 0)
    c = bench.Run("cpp", "fast", _summary("cpp", [[10, 1]], [100], samples=999), 0)
    bench.check_agreement([a, a])
    with pytest.raises(SystemExit, match="rust disagrees with python on decisions"):
        bench.check_agreement([a, b])
    with pytest.raises(SystemExit, match="cpp disagrees with python on samples"):
        bench.check_agreement([a, c])


def test_table_rows_report_latency_in_microseconds_and_memory_in_mib():
    fast = bench.Run("rust", "fast", _summary("rust", [[1, 1]] * 4, [1000, 2000, 3000, 40000], samples=500_000, elapsed=250_000_000), 3 * 2**20)
    paced = bench.Run("cpp", "paced", _summary("cpp", [[1, 1]], [7500], paced=True), 2**20)
    table = bench.table([fast, paced])
    lines = table.splitlines()
    assert lines[0].startswith("| Implementation | Mode | Decisions |")
    assert lines[2] == "| Rust | fast | 4 | 2.0 | n/a | 40.0 | 2,000,000 | 3.0 |"  # p99 needs 100 decisions
    assert lines[3] == "| C++ | paced | 1 | 7.5 | n/a | 7.5 | n/a | 1.0 |"
    many = bench.Run("rust", "fast", _summary("rust", [[1, 1]] * 100, list(range(1000, 101000, 1000))), 0)
    assert bench.table([many]).splitlines()[2] == "| Rust | fast | 100 | 50.0 | 99.0 | 100.0 | 500,000 | 0.0 |"


def test_caption_describes_only_the_modes_that_ran_at_the_models_rate():
    fast = bench.Run("rust", "fast", _summary("rust", [[1, 1]] * 100, [1000] * 100), 0)
    paced = bench.Run("rust", "paced", {**_summary("rust", [[24, 1], [49, 2], [60, 1]], [1, 2, 3], paced=True), "chunk": 25}, 0)
    only_fast = bench.report([fast], GOLDEN_DIR / "stream.csv", 20, 500.0)
    assert "at 500 Hz" in only_fast and "**fast**" in only_fast and "**paced**" not in only_fast
    assert "reads the clock once per sample" in only_fast and "n/a below" not in only_fast
    only_paced = bench.report([paced], GOLDEN_DIR / "stream.csv", 20, 250.0)
    assert "**fast**" not in only_paced and "25-sample chunks delivered at 250 Hz" in only_paced
    assert "2 of 3 decisions fall on the last sample of their chunk" in only_paced
    assert "p99 reads n/a below 100 decisions" in only_paced


def test_a_release_binary_older_than_its_sources_is_stale(tmp_path, monkeypatch):
    binary, src = tmp_path / "mi-replay", tmp_path / "src"
    src.mkdir()
    (src / "lib.rs").write_text("")
    binary.write_text("")
    monkeypatch.setattr(bench, "REPO", tmp_path)
    monkeypatch.setitem(bench.COMMANDS, "rust", [str(binary)])
    monkeypatch.setitem(bench.SOURCES, "rust", [src])
    os.utime(src / "lib.rs", (1000, 1000))
    os.utime(binary, (2000, 2000))
    assert bench.stale("rust") is None
    os.utime(src / "lib.rs", (3000, 3000))
    assert bench.stale("rust") == (f"{binary} is older than src/lib.rs: rebuild it with "
                                   "(cd rust/mi_core && cargo build --release --bin mi-replay)")
    assert bench.stale("python") is None


def test_a_missing_release_binary_names_its_build_command(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(bench.COMMANDS, "cpp", [str(tmp_path / "mi_replay")])
    with pytest.raises(SystemExit):
        bench.main(["--impls", "cpp", "--modes", "fast"])
    assert "is not built: build it with (cd cpp && cmake --preset release && cmake --build --preset release)" \
        in capsys.readouterr().err


def test_readme_block_is_replaced_between_the_markers(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("intro\n<!-- bench:start -->\nold\n<!-- bench:end -->\noutro\n")
    bench.update_readme(readme, "new table")
    assert readme.read_text() == "intro\n<!-- bench:start -->\nnew table\n<!-- bench:end -->\noutro\n"
    readme.write_text("no markers\n")
    with pytest.raises(SystemExit, match="bench:start"):
        bench.update_readme(readme, "x")


def test_run_command_returns_the_summary_and_the_peak_memory():
    run = bench.run_command("python", "fast", [*bench.COMMANDS["python"], str(GOLDEN_DIR / "model.json"),
                                               str(GOLDEN_DIR / "edge_stream.csv"), "--chunk", "1"])
    assert run.summary["impl"] == "python" and len(run.summary["decisions"]) == 9
    assert 10 * 2**20 < run.peak_rss_bytes < 2**32  # a Python process with NumPy: tens of MiB


def test_a_failing_command_stops_the_benchmark():
    with pytest.raises(SystemExit, match="exit 2: usage:"):
        bench.run_command("python", "fast", [sys.executable, "-m", "mi_ref.replay"])


def test_a_command_that_ignores_its_options_is_refused():
    cmd = [*bench.COMMANDS["python"], str(GOLDEN_DIR / "model.json"), str(GOLDEN_DIR / "edge_stream.csv"), "--chunk", "1"]
    with pytest.raises(SystemExit, match="echoed"):
        bench.run_command("python", "paced", cmd, {"impl": "python", "chunk": 1, "paced": True, "repeat": 1})


def test_a_recording_without_decisions_is_refused(tmp_path):
    short = tmp_path / "short.csv"
    short.write_text("".join((GOLDEN_DIR / "edge_stream.csv").read_text().splitlines(keepends=True)[:101]))
    with pytest.raises(SystemExit, match="no decisions"):
        bench.run_command("python", "fast", [*bench.COMMANDS["python"], str(GOLDEN_DIR / "model.json"), str(short)])


def test_main_prints_a_table_for_the_python_reference(capsys):
    assert bench.main(["--impls", "python", "--modes", "fast", "--repeat", "1", "--csv", str(GOLDEN_DIR / "edge_stream.csv")]) == 0
    out = capsys.readouterr().out
    assert "| Python (reference) | fast |" in out
    assert "edge_stream.csv" in out


def test_readme_markers_are_checked_before_any_run(tmp_path, monkeypatch):
    readme = tmp_path / "README.md"
    readme.write_text("no markers\n")
    monkeypatch.setattr(bench, "run_command", lambda *a: pytest.fail("ran before checking the README"))
    with pytest.raises(SystemExit, match="bench:start"):
        bench.main(["--impls", "python", "--modes", "fast", "--readme", str(readme)])


def test_main_writes_the_table_into_the_readme(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("intro\n<!-- bench:start -->\n<!-- bench:end -->\n")
    assert bench.main(["--impls", "python", "--modes", "fast", "--repeat", "1", "--csv", str(GOLDEN_DIR / "edge_stream.csv"), "--readme", str(readme)]) == 0
    text = readme.read_text(encoding="utf-8")
    assert "| Python (reference) | fast | 9 |" in text and text.startswith("intro\n<!-- bench:start -->\nReplay of `edge_stream.csv`")
    assert not (tmp_path / "README.md.tmp").exists()


def test_readme_is_only_written_from_a_golden_recording(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("<!-- bench:start -->\n<!-- bench:end -->\n")
    real = tmp_path / "subject01_session001.csv"
    real.write_bytes((GOLDEN_DIR / "edge_stream.csv").read_bytes())
    with pytest.raises(SystemExit) as e:
        bench.main(["--impls", "python", "--modes", "fast", "--repeat", "1", "--csv", str(real), "--readme", str(readme)])
    assert e.value.code == 2
    assert readme.read_text() == "<!-- bench:start -->\n<!-- bench:end -->\n"
