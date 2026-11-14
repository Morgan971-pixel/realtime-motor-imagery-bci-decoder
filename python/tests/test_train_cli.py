import pytest

from mi_ref import train
from mi_ref.model import Model
from mi_ref.paths import REPO
from mi_ref.recording import save_recording
from mi_ref.synth import make_recording


def _csvs(tmp_path, effect):
    paths = []
    for seed in (0, 1):
        path = tmp_path / f"s{seed}.csv"
        save_recording(path, make_recording(n_trials=20, seed=seed, effect=effect))
        paths.append(str(path))
    return paths


def test_trains_reports_accuracy_and_writes_a_loadable_model(tmp_path, capsys):
    out = tmp_path / "models" / "me.json"
    assert train.main([*_csvs(tmp_path, 0.6), "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "on 40 trials" in text and ": above chance" in text and f"wrote {out}" in text
    model = Model.from_json(out.read_text())
    assert model.fs == 250.0 and model.rail_abs is None


def test_warns_when_the_recordings_are_not_decodable(tmp_path, capsys):
    assert train.main([*_csvs(tmp_path, 0.0), "--out", str(tmp_path / "m.json"), "--rail-abs", "9e5"]) == 0
    assert "NOT above chance" in capsys.readouterr().out
    assert Model.from_json((tmp_path / "m.json").read_text()).rail_abs == 9e5


def test_refuses_an_output_inside_the_repo_or_an_existing_file(tmp_path, capsys):
    csvs = _csvs(tmp_path, 0.6)
    with pytest.raises(SystemExit) as e:
        train.main([*csvs, "--out", str(REPO / "public-data" / "m.json")])
    assert e.value.code == 2 and "stay outside the repository" in capsys.readouterr().err
    (tmp_path / "old.json").write_text("{}")
    with pytest.raises(SystemExit) as e:
        train.main([*csvs, "--out", str(tmp_path / "old.json")])
    assert e.value.code == 2 and "already exists" in capsys.readouterr().err


def test_a_bad_recording_is_reported(tmp_path, capsys):
    bad = tmp_path / "bad.csv"
    bad.write_text("not,a,recording\n")
    assert train.main([str(bad), "--out", str(tmp_path / "m.json")]) == 1
    assert "unexpected columns" in capsys.readouterr().err
    assert not (tmp_path / "m.json").exists()
