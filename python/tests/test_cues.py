import os
import uuid

import pytest

from mi_ref import cues
from mi_ref.cues import MIN_IMAGERY_S, Cue, present, schedule
from mi_ref.events import imagery_markers, markers_from_stim


def test_schedule_is_balanced_timed_and_uses_club_codes():
    s = schedule(n_trials=4, seed=1, prepare_s=3, imagery_s=5, rest_s=3, lead_in_s=2)
    assert [c.t for c in s] == [2, 5, 10, 13, 16, 21, 24, 27, 32, 35, 38, 43]
    assert [c.code // 100 for c in s] == [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
    assert [(c.code // 10) % 10 for c in s] == [2, 3, 5] * 4
    moves = [c.code % 10 for c in s[1::3]]
    assert sorted(moves) == [1, 1, 2, 2]
    assert all(c.code % 10 == moves[i // 3] for i, c in enumerate(s))  # one hand per trial
    assert s[1].text == f"IMAGINE {'LEFT' if moves[0] == 1 else 'RIGHT'} HAND"


def test_schedule_is_reproducible_per_seed_and_shuffled():
    assert schedule(40, seed=7) == schedule(40, seed=7)
    orders = {tuple(c.code % 10 for c in schedule(40, seed=k)[1::3]) for k in range(5)}
    assert len(orders) == 5


def test_imagery_codes_drive_the_decoder_like_the_club_files():
    s = schedule(6, seed=0)
    stim = [0] * 3000
    for i, c in enumerate(s):
        stim[i * 100 + 5] = c.code
    found = imagery_markers(markers_from_stim(stim))
    assert [(m.trial, m.movement) for m in found] == [(c.code // 100, c.code % 10) for c in s[1::3]]


@pytest.mark.parametrize("kwargs,message", [
    ({"n_trials": 3}, "even number"), ({"n_trials": 0}, "even number"),
    ({"imagery_s": MIN_IMAGERY_S - 0.01}, "at least 2.5 s"), ({"prepare_s": 0}, "prepare time"),
    ({"rest_s": float("inf")}, "rest time"), ({"lead_in_s": -1}, "lead-in"),
])
def test_schedule_refuses_bad_settings(kwargs, message):
    with pytest.raises(ValueError, match=message):
        schedule(**kwargs)


class FakeClock:
    def __init__(self):
        self.now = 50.0
        self.slept = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(round(seconds, 9))
        self.now += seconds + 0.001  # oversleeps a little every time


def test_present_paces_on_absolute_deadlines_and_stamps_each_marker():
    clock, shown, pushed = FakeClock(), [], []

    class Outlet:
        def push_sample(self, sample, timestamp):
            pushed.append((sample, timestamp))

    s = [Cue(0.0, 121, "get ready: LEFT"), Cue(3.0, 131, "IMAGINE LEFT HAND"), Cue(8.0, 151, "rest")]
    assert present(s, Outlet(), local_clock=lambda: clock.now + 1000, show=shown.append,
                   clock=clock, sleep=clock.sleep) == 3
    assert shown == ["get ready: LEFT", "IMAGINE LEFT HAND", "rest"]
    assert clock.slept == [3.0, 4.999]  # oversleep is absorbed, not accumulated
    assert [p[0] for p in pushed] == [[121], [131], [151]]
    assert [round(p[1], 6) for p in pushed] == [1050.0, 1053.001, 1058.001]


def test_main_refuses_bad_settings_before_touching_lsl(capsys):
    with pytest.raises(SystemExit) as e:
        cues.main(["--trials", "5"])
    assert e.value.code == 2
    assert "even number" in capsys.readouterr().err


def test_markers_reach_an_lsl_inlet_as_int32_codes():
    try:
        import pylsl
    except ImportError:
        if os.environ.get("REQUIRE_LSL") == "1":
            raise
        pytest.skip("pylsl is not installed")
    info = cues.marker_info("mi-realtime-cues-test", f"cues-{uuid.uuid4()}", pylsl)
    assert (info.type(), info.channel_count(), info.nominal_srate()) == ("Markers", 1, 0.0)
    outlet = pylsl.StreamOutlet(info)
    inlet = pylsl.StreamInlet(info)
    inlet.open_stream(timeout=2)
    s = schedule(2, lead_in_s=0, prepare_s=0.01, imagery_s=2.5, rest_s=0.01)
    present(s, outlet, pylsl.local_clock, show=lambda text: None, sleep=lambda seconds: None)
    got = []
    while len(got) < len(s):
        chunk, stamps = inlet.pull_chunk(timeout=5)
        assert chunk, f"received {len(got)} of {len(s)} markers"
        got += [(sample[0], stamp) for sample, stamp in zip(chunk, stamps)]
    inlet.close_stream()
    assert [code for code, _ in got] == [c.code for c in s]
    assert all(b >= a for (_, a), (_, b) in zip(got, got[1:]))
