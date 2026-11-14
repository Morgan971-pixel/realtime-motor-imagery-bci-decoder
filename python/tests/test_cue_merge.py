import math

import pytest

from mi_ref.cue_merge import CueMerger, parse_code

FS = 250.0
DT = 1 / FS


def _feed(merger, n, t0=100.0):
    """n EEG samples on an exact grid; returns the released (index, stim) pairs so far."""
    out = []
    for i in range(n):
        merger.add_eeg(i, t0 + i * DT)
        out += merger.pop_ready()
    return out


def _stims(items):
    return [(values, stim) for values, stim, _ in items if stim]


@pytest.mark.parametrize("raw,code", [
    (231, 231), (231.0, 231), ("231", 231), (" 4032 ", 4032), (2**31 - 1, 2**31 - 1),
    (99, None), (0, None), (-231, None), (2**31, None), (231.5, None), (math.nan, None),
    ("2x1", None), ("", None), (True, None), (None, None), ([231], None),
])
def test_parse_code(raw, code):
    assert parse_code(raw) == code


def test_cue_lands_on_the_nearest_sample():
    merger = CueMerger(FS, lag=0.0)
    t0 = 100.0
    merger.add_cue(132, t0 + 10 * DT)               # exactly on sample 10
    merger.add_cue(231, t0 + 20.4 * DT)             # nearer 20
    merger.add_cue(332, t0 + 30.6 * DT)             # nearer 31
    merger.add_cue(431, t0 + 40.49 * DT)            # just short of halfway: 40
    merger.add_cue(532, t0 + 50 * DT - 1e-9)        # a hair before 50 (smoothed timestamps): still 50
    released = _feed(merger, 60, t0) + merger.flush()
    assert _stims(released) == [(10, 132), (20, 231), (31, 332), (40, 431), (50, 532)]
    assert (merger.late, merger.invalid, merger.collisions) == (0, 0, 0)


def test_samples_wait_for_the_lag_then_release_in_order():
    merger = CueMerger(FS, lag=0.1)  # 25 samples
    released = _feed(merger, 30)
    assert [values for values, _, _ in released] == list(range(5))  # samples 0..4 are >= 0.1 s old
    assert [values for values, _, _ in merger.flush()] == list(range(5, 30))


def test_a_cue_arriving_within_the_lag_still_lands():
    merger = CueMerger(FS, lag=0.1)
    released = _feed(merger, 20)  # nothing released yet: newest is sample 19, 0.076 s
    assert released == []
    merger.add_cue(231, 100.0 + 3 * DT)  # arrives after sample 3 was received, before its release
    assert _stims(merger.pop_ready() + merger.flush()) == [(3, 231)]
    assert merger.late == 0


def test_a_late_cue_is_counted_reported_and_never_moved():
    messages = []
    merger = CueMerger(FS, lag=0.0, report=messages.append)
    released = _feed(merger, 10)
    merger.add_cue(231, 100.0 + 5 * DT)  # sample 5 already released
    released += merger.flush()
    assert _stims(released) == []
    assert merger.late == 1
    assert "dropped late cue 231" in messages[0] and "--cue-lag" in messages[0]


def test_invalid_cues_and_a_second_cue_on_one_sample_are_dropped():
    messages = []
    merger = CueMerger(FS, lag=0.0, report=messages.append)
    merger.add_cue("left", 100.0)
    merger.add_cue(42, 100.0)
    merger.add_cue(231, 100.0 + 2 * DT)
    merger.add_cue(331, 100.0 + 2.2 * DT)  # also nearest sample 2
    released = _feed(merger, 5) + merger.flush()
    assert _stims(released) == [(2, 231)]
    assert (merger.invalid, merger.collisions) == (2, 1)
    assert messages[0] == "dropped invalid cue 'left' at 100.000000 s"
    assert messages[2] == "dropped cue 331: cue 231 is already on the same EEG sample"


def test_cues_arriving_out_of_order_are_sorted_by_time():
    merger = CueMerger(FS, lag=1.0)
    merger.add_cue(331, 100.0 + 30 * DT)
    merger.add_cue(231, 100.0 + 10 * DT)
    assert _stims(_feed(merger, 40) + merger.flush()) == [(10, 231), (30, 331)]


def test_equal_eeg_timestamps_are_allowed_but_not_backwards():
    merger = CueMerger(FS)
    merger.add_eeg(0, 1.0)
    merger.add_eeg(1, 1.0)  # bursty raw timestamps repeat
    with pytest.raises(ValueError, match="went backwards"):
        merger.add_eeg(2, 0.9)
    with pytest.raises(ValueError, match="must be finite"):
        merger.add_eeg(3, math.inf)


@pytest.mark.parametrize("fs,lag,message", [(0, 0.5, "fs must be"), (math.inf, 0.5, "fs must be"),
                                            (250, -1, "cue lag"), (250, math.nan, "cue lag")])
def test_bad_settings_are_refused(fs, lag, message):
    with pytest.raises(ValueError, match=message):
        CueMerger(fs, lag)


@pytest.mark.parametrize("raw", ["²", "٣١", "1" * 11])
def test_unicode_digits_and_huge_strings_are_invalid_not_a_crash(raw):
    assert parse_code(raw) is None


def test_cues_left_after_the_last_sample_are_counted_and_reported():
    messages = []
    merger = CueMerger(FS, lag=0.0, report=messages.append)
    merger.add_cue(231, 100.0 + 50 * DT)  # after the last sample below
    _feed(merger, 10)
    assert _stims(merger.flush()) == []
    assert merger.unmatched == 1
    assert messages == ["dropped 1 cue(s) after the last EEG sample: [231]"]
