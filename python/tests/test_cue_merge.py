import bisect
import math
from dataclasses import dataclass

import pytest
from hypothesis import event, given
from hypothesis import strategies as st

from mi_ref.cue_merge import CueMerger, parse_code
from mi_ref.events import MAX_CODE

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


# Property tests over generated live-stream scenarios.
#
# The oracle works in integer ticks of 10 ns, so every comparison in the documented rules is exact.
# Sample timestamps sit on whole microseconds after t0, and 1/fs and half of it are whole
# microseconds for every fs below. Every cue's (timestamp - half sample) and every nonzero lag sit
# 0.37 us off that grid, so no cue is within rounding of exactly halfway between samples (where
# the module allows either neighbour) and no sample is within rounding of exactly `lag` old.

TICKS = 10**8  # ticks per second
US = 100  # ticks per microsecond
OFF = 37  # ticks off the microsecond grid
FS_CHOICES = (125.0, 200.0, 250.0, 500.0, 1000.0)
VALID, BAD_CODE, BAD_TIME = "valid", "bad code", "bad time"


@dataclass
class Cue:
    raw: object
    code: int | None  # what the cue should parse to; None if its value is not a valid code
    kind: str
    ticks: int  # cue timestamp, ticks after t0 (unused for a non-finite timestamp)
    slot: int  # arrives after this many EEG samples have been added


@dataclass
class Scenario:
    fs: float
    steady: bool
    t0: int  # ticks
    ticks: list  # sample timestamps, ticks after t0, non-decreasing
    lag: int  # ticks
    cues: list
    steps: list  # per sample: "pop, cues", "cues, pop" or "cues" (no pop) after it is added

    def seconds(self, ticks):
        return (self.t0 + ticks) / TICKS


_codes = st.integers(100, MAX_CODE)


@st.composite
def _digit_string(draw):
    code = draw(_codes)
    zeros = min(draw(st.integers(0, 3)), 10 - len(str(code)))
    pad = draw(st.sampled_from(["", " ", "\t"])), draw(st.sampled_from(["", " ", "\n"]))
    return pad[0] + "0" * zeros + str(code) + pad[1], code


_valid_raw = st.one_of(
    _codes.map(lambda c: (c, c)),
    _digit_string(),
    _codes.map(lambda c: (float(c), c)),
)

_invalid_raw = st.one_of(
    st.one_of(st.integers(max_value=99), st.integers(min_value=MAX_CODE + 1)),
    st.one_of(st.integers(0, 99), st.integers(MAX_CODE + 1, 10**10 - 1)).map(str),
    st.floats().filter(lambda x: not (math.isfinite(x) and x.is_integer() and 100 <= x <= MAX_CODE)),
    st.sampled_from([True, False, None, [231], b"231", "left", "", " ", "+231", "-231", "2 31", "231.0",
                     "²", "٣١", "1" * 11]),
    # any character other than whitespace or an ASCII digit makes a string invalid
    st.text(max_size=6).filter(lambda s: s.strip() == "" or any(not c.isspace() and c not in "0123456789" for c in s)),
)


@st.composite
def _cues(draw, ticks, period, half):
    n = len(ticks)
    cues, anchors = [], []
    for _ in range(draw(st.integers(0, 12))):
        kind = draw(st.sampled_from([VALID, VALID, VALID, VALID, BAD_CODE, BAD_TIME]))
        if kind == BAD_CODE:
            raw, code = draw(_invalid_raw), None
        else:
            raw, code = draw(_valid_raw)
        # near any sample, near an earlier cue (a collision) or near the last sample (unmatched)
        anchor = draw(st.one_of(st.integers(0, n - 1), st.just(n - 1), *[st.sampled_from(anchors)] * bool(anchors)))
        anchors.append(anchor)
        off_us = draw(st.integers(-3 * period // US, 3 * period // US))
        cue_ticks = ticks[anchor] + half + off_us * US + OFF  # (cue - half) is near sample `anchor`
        # before its sample, soon after it (within a typical lag) or long after it (late)
        delay = draw(st.one_of(st.integers(-20, 10), st.integers(-20, 120)))
        cues.append(Cue(raw, code, kind, cue_ticks, min(max(anchor + 1 + delay, 0), n)))
    return cues


@st.composite
def scenarios(draw):
    fs = draw(st.sampled_from(FS_CHOICES))
    period = round(TICKS / fs)
    half = period // 2
    n = draw(st.integers(1, 200))
    steady = draw(st.booleans())
    if steady:
        ticks = [i * period for i in range(n)]
    else:  # bursty: repeated stamps, short gaps and stalls of up to 0.4 s
        gap_us = st.one_of(st.just(0), st.integers(1, 2 * period // US), st.integers(1, 400_000))
        gaps = draw(st.lists(gap_us, min_size=n - 1, max_size=n - 1))
        ticks = [0]
        for g in gaps:
            ticks.append(ticks[-1] + g * US)
    t0 = draw(st.integers(0, 10**11)) * US  # up to 1e5 s, like a local LSL clock
    lag = draw(st.one_of(st.just(0), st.integers(0, 300_000).map(lambda u: u * US + OFF)))
    cues = draw(_cues(ticks, period, half))
    steps = draw(st.lists(st.sampled_from(["pop, cues", "cues, pop", "cues"]), min_size=n, max_size=n))
    return Scenario(fs, steady, t0, ticks, lag, cues, steps)


def _run(scn):
    """Drive a CueMerger through the scenario as a live stream would, and the oracle alongside it.

    Returns what the merger did and what the documented rules say it should do."""
    half = round(TICKS / scn.fs) // 2
    n = len(scn.ticks)
    values = [[i] for i in range(n)]  # distinct objects, so identity shows each sample's own values
    messages = []
    merger = CueMerger(scn.fs, lag=scn.lag / TICKS, report=messages.append)
    batches, expected_batches = [], []
    added = released = 0
    late = invalid = arrivals = 0
    waiting = []  # (cue ticks, arrival order, code, target) of valid cues accepted by the oracle

    def pop():
        nonlocal released
        batches.append(merger.pop_ready())
        # documented: release every held sample at least `lag` older than the newest sample
        start = released
        while released < added and scn.ticks[released] <= scn.ticks[added - 1] - scn.lag:
            released += 1
        expected_batches.append(list(range(start, released)))

    def cues_at(slot):
        nonlocal late, invalid, arrivals
        for cue in scn.cues:
            if cue.slot != slot:
                continue
            timestamp = math.nan if cue.kind == BAD_TIME else scn.seconds(cue.ticks)
            merger.add_cue(cue.raw, timestamp)
            if cue.kind != VALID:
                invalid += 1
                continue
            # documented: the first sample whose timestamp is at least the cue's minus half a period
            target = bisect.bisect_left(scn.ticks, cue.ticks - half)
            if target < released:  # its sample was already released: late, never moved
                late += 1
            else:  # ties go to the cue that ARRIVED first, not the one listed first in the scenario
                waiting.append((cue.ticks, arrivals, cue.code, target))
                arrivals += 1

    cues_at(0)
    for i in range(n):
        merger.add_eeg(values[i], scn.seconds(scn.ticks[i]))
        added += 1
        for action in scn.steps[i].split(", "):
            pop() if action == "pop" else cues_at(i + 1)
    batches.append(merger.flush())
    expected_batches.append(list(range(released, n)))

    stims = [0] * n
    collisions = unmatched = 0
    for cue_ticks, _, code, target in sorted(waiting):  # earliest cue first, ties by arrival
        if target == n:
            unmatched += 1
        elif stims[target]:
            collisions += 1
        else:
            stims[target] = code
    if scn.steady:  # on a steady stream the rule is the nearest sample, as the docs say
        for cue_ticks, _, _, target in waiting:
            if target < n:
                assert target == min(range(n), key=lambda i: abs(scn.ticks[i] - cue_ticks))

    expected = dict(stims=stims, late=late, invalid=invalid, collisions=collisions, unmatched=unmatched,
                    placed=sum(1 for s in stims if s), batches=expected_batches)
    for name, count in (("late", late), ("invalid", invalid), ("collisions", collisions),
                        ("unmatched", unmatched), ("placed", expected["placed"])):
        event(f"{name}: {'some' if count else 'none'}")
    event("stream: steady" if scn.steady else "stream: bursty")
    return merger, values, batches, messages, expected


@given(scenarios())
def test_every_sample_comes_out_once_in_order_with_its_own_values(scn):
    merger, values, batches, _, expected = _run(scn)
    out = [item for batch in batches for item in batch]
    assert len(out) == len(scn.ticks)
    for i, (v, _, timestamp) in enumerate(out):
        assert v is values[i]
        assert timestamp == scn.seconds(scn.ticks[i])
    # and each batch holds exactly the samples the lag rule releases at that call
    assert [[v[0] for v, _, _ in batch] for batch in batches] == expected["batches"]


@given(scenarios())
def test_every_cue_lands_on_its_sample_or_is_counted_as_dropped(scn):
    merger, _, batches, messages, expected = _run(scn)
    stims = [stim for batch in batches for _, stim, _ in batch]
    assert stims == expected["stims"]
    assert (merger.late, merger.invalid, merger.collisions, merger.unmatched) == (
        expected["late"], expected["invalid"], expected["collisions"], expected["unmatched"])
    # one report per dropped cue, except unmatched cues, which share one at flush
    assert len(messages) == merger.late + merger.invalid + merger.collisions + bool(merger.unmatched)
    assert (merger.summary() is None) == (len(messages) == 0)


@given(scenarios())
def test_placed_and_dropped_cues_add_up_to_the_cues_added(scn):
    merger, _, batches, _, _ = _run(scn)
    placed = sum(1 for batch in batches for _, stim, _ in batch if stim)
    dropped = merger.late + merger.invalid + merger.collisions + merger.unmatched
    assert placed + dropped == len(scn.cues)
