import numpy as np
import pytest
from scipy.stats import binom

from mi_ref import evaluate
from mi_ref.evaluate import chance_threshold, contiguous_folds, cross_validate, main, usable_channels
from mi_ref.model import Model, decode_recording, train
from mi_ref.recording import Recording, save_recording
from mi_ref.synth import make_recording


def test_chance_threshold_is_smallest_significant_accuracy():
    for n in (32, 64, 100):
        k = round(chance_threshold(n) * n)
        assert binom.sf(k - 1, n, 0.5) <= 0.05  # P(X >= k) small enough
        assert binom.sf(k - 2, n, 0.5) > 0.05  # one fewer is not


def test_contiguous_folds_keep_groups_whole_and_in_time_order():
    groups = np.repeat(np.arange(10), 4)
    folds = contiguous_folds(groups, 5)
    assert sorted(np.concatenate(folds)) == list(range(40))
    for f, test in enumerate(folds):
        assert list(np.unique(groups[test])) == [2 * f, 2 * f + 1]  # neighbours in time, held out together


def test_cross_validate_finds_real_signal():
    r = cross_validate([make_recording(n_trials=40, seed=0, effect=0.6)])
    assert r["n_trials"] == 40
    assert r["accuracy"] >= 0.85


def test_cross_validate_stays_at_chance_without_signal():
    """Null control over 10 seeds, so it does not hinge on one lucky draw."""
    results = [cross_validate([make_recording(n_trials=40, seed=s, effect=0.0)]) for s in range(10)]
    accs = [r["accuracy"] for r in results]
    assert abs(np.mean(accs) - 0.5) < 0.1
    assert np.mean([r["accuracy"] >= r["chance_95"] for r in results]) <= 0.3


def test_two_recordings_with_the_same_trial_numbers_are_separate_groups():
    r = cross_validate([make_recording(n_trials=20, seed=1), make_recording(n_trials=20, seed=2)])
    assert r["n_groups"] == 40


def test_a_single_group_is_a_clear_error():
    with pytest.raises(ValueError, match="at least two trial groups"):
        cross_validate([make_recording(n_trials=1, seed=0)])


def test_main_reports_subjects_and_skips(tmp_path, capsys):
    good = tmp_path / "subjectA" / "session001"
    short = tmp_path / "subjectB" / "session001"
    good.mkdir(parents=True)
    short.mkdir(parents=True)
    save_recording(good / "rec.csv", make_recording(n_trials=40, seed=0))
    save_recording(short / "rec.csv", make_recording(n_trials=4, seed=0))
    assert main(["--data-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "1 recording(s) skipped" in out
    assert "subjectA" in out and "subjectB" not in out
    assert "expected by luck alone" in out


def test_main_with_no_usable_subjects_fails(tmp_path, capsys):
    assert main(["--data-dir", str(tmp_path)]) == 1


RAIL = 9e5  # above every synthetic offset (2e5 to 8e5)


def _pin(rec, channels, fraction=1.0):
    """Drive channels to just past RAIL for the first `fraction` of the recording."""
    signal = rec.signal.copy()
    n = int(round(fraction * len(signal)))
    signal[:n, channels] = RAIL + 1e4 + rec.signal[:n, channels] - rec.signal[:n, channels].mean(axis=0)
    return Recording(rec.timestamps, signal, rec.stim)


def test_usable_channels_drops_channels_pinned_half_the_time_or_more():
    rec = make_recording(n_trials=4, seed=0)
    rec = _pin(_pin(rec, [2], 0.5), [6], 0.4)
    other = make_recording(n_trials=4, seed=1)
    assert list(usable_channels([rec], RAIL)) == [0, 1, 3, 4, 5, 6, 7]  # Cz at 50% goes, Oz at 40% stays
    assert list(usable_channels([rec, other], RAIL)) == [0, 1, 2, 3, 4, 5, 6, 7]  # Cz now 25% overall


def test_training_without_a_channel_gives_it_zero_weight_and_the_decoder_ignores_it():
    rec = make_recording(n_trials=40, seed=0)
    keep = [0, 1, 3, 4, 5, 6, 7]
    model = train([rec], keep=keep, rail_abs=RAIL)
    assert all(np.all(W[:, 2] == 0.0) for W in model.csp)
    assert model.used_channels == tuple(keep) and model.rail_abs == RAIL
    again = Model.from_json(model.to_json())
    assert again.used_channels == tuple(keep)
    pinned = _pin(make_recording(n_trials=10, seed=1), [2])  # Cz stuck: no decision is refused
    decisions = decode_recording(again, pinned)
    assert all(d["pinned"] == [] for d in decisions)
    assert np.mean([d["label"] == d["true_label"] for d in decisions]) >= 0.8


def test_cross_validate_on_kept_channels_still_finds_the_signal(monkeypatch):
    """The stuck channels carry only noise; the signal is on the kept ones. Every fold's CSP must
    give the dropped channels exactly-zero columns, so they never reach the fit."""
    rec = make_recording(n_trials=40, seed=0, effect=0.6)
    signal = rec.signal.copy()
    noise = np.random.default_rng(5).normal(0.0, 20.0, size=(len(signal), 2))
    signal[:, [2, 6]] = RAIL + 1e4 + noise  # stuck at the rail, no EEG
    rec = Recording(rec.timestamps, signal, rec.stim)
    keep = usable_channels([rec], RAIL)
    assert list(keep) == [0, 1, 3, 4, 5, 7]
    fitted, real = [], evaluate.fit_epochs

    def spy(*args, **kwargs):
        csp, w, b = real(*args, **kwargs)
        fitted.append(csp)
        return csp, w, b

    monkeypatch.setattr(evaluate, "fit_epochs", spy)
    r = cross_validate([rec], keep=keep)
    assert r["accuracy"] >= 0.85
    assert len(fitted) == 5
    assert all(np.all(W[:, [2, 6]] == 0.0) for csp in fitted for W in csp)


@pytest.mark.parametrize("keep", [[-1, 0, 1, 2], [0, 1, 2, 8], [0, 1, 2], [0, 1, 1, 2]])
def test_train_refuses_a_bad_keep(keep):
    with pytest.raises(ValueError, match="keep"):
        train([make_recording(n_trials=8, seed=0)], keep=keep)


@pytest.mark.parametrize("keep", [[True, True, True, True], [True, 0, 2, 3], [np.True_, 0, 2, 3],
                                  [0.0, 1.0, 2.0, 3.0], [0, 1, 2.5, 3], ["0", "1", "2", "3"]])
def test_train_refuses_keep_that_is_not_integers(keep):
    with pytest.raises(ValueError, match="integer channel indices"):
        train([make_recording(n_trials=8, seed=0)], keep=keep)


def test_train_reports_an_empty_keep_as_too_few_channels():
    with pytest.raises(ValueError, match="at least 4 channels"):
        train([make_recording(n_trials=8, seed=0)], keep=[])


def test_train_dedupes_and_sorts_keep():
    recs = [make_recording(n_trials=8, seed=0)]
    model = train(recs, keep=[3, 1, 0, 2, 3])
    assert model.used_channels == (0, 1, 2, 3)
    plain = train(recs, keep=[0, 1, 2, 3])
    assert all(np.array_equal(W, V) for W, V in zip(model.csp, plain.csp))


def test_evaluate_minimum_channels_follows_the_csp_pairs():
    assert evaluate.MIN_CHANNELS == 2 * evaluate.N_PAIRS


def test_main_exclude_pinned_reports_dropped_channels_and_skips_hopeless_subjects(tmp_path, capsys):
    one = tmp_path / "subjectA" / "session001"
    many = tmp_path / "subjectB" / "session001"
    one.mkdir(parents=True)
    many.mkdir(parents=True)
    save_recording(one / "rec.csv", _pin(make_recording(n_trials=40, seed=0), [2]))
    save_recording(many / "rec.csv", _pin(make_recording(n_trials=40, seed=1), [0, 2, 4, 5, 6]))
    assert main(["--data-dir", str(tmp_path), "--exclude-pinned", str(RAIL)]) == 0
    out = capsys.readouterr().out
    assert "dropped Cz" in out.splitlines()[2]
    assert "subjectB     skipped: only 3 usable channels" in out


@pytest.mark.parametrize("rail", ["0", "-5", "inf", "nan"])
def test_main_exclude_pinned_needs_a_positive_finite_rail(tmp_path, rail):
    with pytest.raises(SystemExit) as e:
        main(["--data-dir", str(tmp_path), "--exclude-pinned", rail])
    assert e.value.code == 2


def test_main_takes_the_rate_and_trial_minimum_from_the_command_line(tmp_path, capsys, monkeypatch):
    rec_dir = tmp_path / "subjectA" / "session001"
    rec_dir.mkdir(parents=True)
    save_recording(rec_dir / "rec.csv", make_recording(n_trials=16, seed=1, fs=160.0))
    assert main(["--data-dir", str(tmp_path)]) == 1  # 16 trials is below the default 30
    assert "1 recording(s) skipped for having fewer than 30 imagery trials" in capsys.readouterr().out
    seen = []
    real = evaluate.cross_validate
    monkeypatch.setattr(evaluate, "cross_validate", lambda recs, fs, keep: seen.append(fs) or real(recs, fs=fs, keep=keep))
    assert main(["--data-dir", str(tmp_path), "--fs", "160", "--min-trials", "10"]) == 0
    out = capsys.readouterr().out
    assert "0 recording(s) skipped for having fewer than 10 imagery trials" in out
    assert "subjectA" in out and seen == [160.0]


@pytest.mark.parametrize("args,message", [
    (["--fs", "0"], "--fs needs a finite rate > 0"), (["--fs", "-160"], "--fs needs a finite rate > 0"),
    (["--fs", "inf"], "--fs needs a finite rate > 0"), (["--fs", "nan"], "--fs needs a finite rate > 0"),
    (["--min-trials", "0"], "--min-trials needs N >= 1"), (["--min-trials", "-3"], "--min-trials needs N >= 1"),
])
def test_main_refuses_a_bad_rate_or_trial_minimum(tmp_path, capsys, args, message):
    with pytest.raises(SystemExit) as e:
        main(["--data-dir", str(tmp_path), *args])
    assert e.value.code == 2
    assert message in capsys.readouterr().err
