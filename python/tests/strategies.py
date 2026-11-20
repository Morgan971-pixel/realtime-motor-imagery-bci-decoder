"""Hypothesis generators for decoder models and streams, shared by the property tests.

models(realistic) builds a Model straight from the contract's valid ranges (docs/model-format.md),
not by training, so it reaches corners training never would: odd windows, zero CSP columns, raw
filter sections. streams(model) builds samples and stim codes for that model, short enough that
hundreds of examples stay cheap and a failure shrinks to a few samples.

realistic=True keeps every value within what a headset produces (about +-1e6) and every filter
stable: the range where the Python reference and the native cores must agree within tolerance.
realistic=False adds extreme values and possibly unstable filters: the range where only Rust and
C++ must agree, bit for bit.

streams also mixes in the hard cases (repeated and overlapping cues, reused trial numbers, prepare
and rest codes, channels at the rail), and streams(..., bad_rows=True) inserts rows the decoders
must reject without changing state, returning their indices as a third value.
"""
import numpy as np
from hypothesis import strategies as st

from mi_ref.epochs import to_samples
from mi_ref.filters import design_band
from mi_ref.model import Model
from mi_ref.recording import EEG_CHANNELS

N_CH = len(EEG_CHANNELS)
MAX_EXTRA = 600  # samples after the longest decision a stream may run on


def _floats(lo, hi):
    return st.floats(lo, hi, allow_nan=False, allow_infinity=False)


@st.composite
def _band(draw, fs, realistic):
    low = draw(_floats(0.02, 0.6)) * fs / 2
    high = low + draw(_floats(0.02, 0.35)) * fs / 2  # stays below fs / 2
    if realistic or draw(st.booleans()):
        sos = design_band(low, high, fs)
    else:  # raw sections with a0 = 1, possibly unstable
        k = draw(st.integers(1, 4))
        coef = draw(st.lists(st.lists(_floats(-2.0, 2.0), min_size=5, max_size=5), min_size=k, max_size=k))
        sos = np.array([[b0, b1, b2, 1.0, a1, a2] for b0, b1, b2, a1, a2 in coef])
    return (low, high), sos


@st.composite
def models(draw, realistic=True):
    """A valid Model: 1-4 bands, 2, 4 or 6 CSP rows with some columns exactly 0.0 or -0.0, a 2-64
    sample window, rail check on or off."""
    fs = draw(_floats(50.0, 1000.0))
    n_bands = draw(st.integers(1, 4))
    bands, sos = zip(*[draw(_band(fs, realistic)) for _ in range(n_bands)])
    rows = draw(st.sampled_from([2, 4, 6]))
    weight = _floats(-3.0, 3.0) if realistic else st.one_of(_floats(-3.0, 3.0), _floats(-1e100, 1e100))
    zero_cols = draw(st.sets(st.integers(0, N_CH - 1), max_size=N_CH))  # all 8: a model that uses no channel
    zero = draw(st.sampled_from([0.0, -0.0]))
    csp = []
    for _ in range(n_bands):
        W = np.array(draw(st.lists(st.lists(weight, min_size=N_CH, max_size=N_CH), min_size=rows, max_size=rows)))
        W[:, sorted(zero_cols)] = zero
        csp.append(W)
    lda = _floats(-10.0, 10.0) if realistic else st.one_of(_floats(-10.0, 10.0), _floats(-1e10, 1e10))
    start, length = draw(st.integers(0, 40)), draw(st.integers(2, 64))
    rail = draw(st.one_of(st.none(), _floats(1e2, 1e6) if realistic else _floats(1.0, 1e300)))
    return Model(fs=fs, channels=EEG_CHANNELS, window_start_s=start / fs, window_len_s=length / fs,
                 bands=bands, sos=list(sos), csp=csp,
                 lda_w=draw(st.lists(lda, min_size=n_bands * rows, max_size=n_bands * rows)), lda_b=draw(lda),
                 rail_abs=rail)


def decision_lag(model) -> int:
    """Samples from an imagery onset to its decision, inclusive of the onset sample."""
    return to_samples(model.window_start_s, model.fs) + to_samples(model.window_len_s, model.fs)


def imagery_codes():
    """Codes that start an imagery trial: trial * 100 + 30 + movement (1 left, 2 right)."""
    return st.builds(lambda trial, movement: trial * 100 + 30 + movement, st.integers(1, 99), st.sampled_from([1, 2]))


def any_codes():
    """Any valid stim code: other phases and movements too, which the decoders must ignore."""
    return st.one_of(imagery_codes(), st.integers(100, 9999), st.integers(100, 2**31 - 1))


def _phase_codes(trial):
    """Imagery (3), prepare (2) and rest (5) codes, often reusing trial: repeated trial numbers."""
    trials = st.one_of(st.just(trial), st.integers(1, 99))
    return st.builds(lambda t, p, m: t * 100 + p * 10 + m, trials, st.sampled_from([3, 2, 5]), st.sampled_from([1, 2]))


def _imagery_with_trial(trial):
    """An imagery code, often for the same trial as another cue."""
    trials = st.one_of(st.just(trial), st.integers(1, 99))
    return st.builds(lambda t, m: t * 100 + 30 + m, trials, st.sampled_from([1, 2]))


# Rows a decoder must reject (docs/model-format.md, Inputs and step 4).
BAD_STIM = (-1.0, -131.0, 1.0, 99.0, 313.5, 131.25, 2.0**31, 1e300, np.nan, np.inf, -np.inf)
BAD_EEG = (np.nan, np.inf, -np.inf)


@st.composite
def _bad_row(draw, model, row, code):
    """(eeg, stim) for one rejected row. Its other values would change the decoder if it took the
    row: a copy of a neighbour, zeros (below any rail, and a different first sample), the rail
    itself, or, with a bad EEG value, an imagery code that would schedule a decision."""
    looks = ["copy", "zeros"] + (["rail"] if model.rail_abs is not None else [])
    look = draw(st.sampled_from(looks))
    if look == "copy":
        eeg = np.array(row, dtype=np.float64)
    elif look == "zeros":
        eeg = np.zeros(N_CH)
    else:
        eeg = np.full(N_CH, draw(st.sampled_from([1.0, -1.0])) * model.rail_abs)
    if draw(st.booleans()):
        return eeg, draw(st.sampled_from(BAD_STIM))
    eeg[draw(st.integers(0, N_CH - 1))] = draw(st.sampled_from(BAD_EEG))
    return eeg, float(draw(st.one_of(st.just(0), st.just(int(code)), imagery_codes())))


@st.composite
def streams(draw, model, realistic=True, bad_rows=False):
    """(signal (n, 8), stim (n,) int64) for model, or (signal, stim float64, bad) with bad_rows.

    One imagery cue with room for its decision, plus up to 5 more cues of any kind at random
    samples. Hard cases, each chosen by Hypothesis: every cue may last 1 to 3 samples; up to 3
    imagery cues within decision_lag(model) of the first (overlapping windows); prepare (2), rest
    (5) and imagery codes that reuse a trial number; and, when model.rail_abs is set, stretches
    where some channels sit at or beyond +-rail_abs, sometimes over the whole window of the first
    decision (the pinned refusal), sometimes with a one-sample dip. EEG comes from a seeded
    generator around a per-channel offset.

    bad_rows=True inserts 1 to 4 rows the decoder must reject (an invalid stim value, or NaN or
    +-inf in one channel) and also returns bad, the sorted row indices of those rows: deleting
    them gives the stream without bad rows, which the decoders must treat identically."""
    lag = decision_lag(model)
    L = to_samples(model.window_len_s, model.fs)
    n = lag + draw(st.integers(0, MAX_EXTRA))
    stim = np.zeros(n, dtype=np.int64)
    g = draw(st.integers(0, n - lag))
    first = draw(imagery_codes())

    def cue(p, code):  # a run of 1 to 3 samples of the same code
        stim[p : p + draw(st.integers(1, 3))] = code

    for p in draw(st.lists(st.integers(max(0, g - lag + 1), min(n - 1, g + lag - 1)), max_size=3)):
        cue(p, draw(_imagery_with_trial(first // 100)))
    for p in draw(st.lists(st.integers(0, n - 1), max_size=5, unique=True)):
        cue(p, draw(st.one_of(any_codes(), _phase_codes(first // 100))))
    cue(g, first)  # last, so nothing overwrites it: at least one decision

    big = 1e6 if realistic else 1e300
    offset = np.array(draw(st.lists(_floats(-big, big), min_size=N_CH, max_size=N_CH)))
    scale = draw(_floats(1e-3, 1e3) if realistic else st.one_of(_floats(1e-3, 1e3), _floats(1e3, 1e300)))
    rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1)))
    signal = offset + scale * rng.standard_normal((n, N_CH))

    if model.rail_abs is not None:
        for _ in range(draw(st.integers(0, 3))):
            chans = sorted(draw(st.sets(st.integers(0, N_CH - 1), min_size=1)))
            level = draw(st.sampled_from([1.0, -1.0])) * model.rail_abs * draw(st.one_of(st.just(1.0), _floats(1.0, 2.0)))
            if draw(st.booleans()):  # covers the first decision's window [g + lag - L, g + lag)
                a = draw(st.integers(max(0, g + lag - L - 3), g + lag - L))
                b = min(n, g + lag + draw(st.integers(0, 3)))
            else:
                a = draw(st.integers(0, n - 1))
                b = min(n, a + draw(st.integers(1, 2 * L)))
            signal[a:b, chans] = level
            if draw(st.integers(0, 3)) == 0:  # one sample below the rail resets the run
                signal[draw(st.integers(a, b - 1)), chans] = 0.0
    if not bad_rows:
        return signal, stim

    # insert before clean row p; g + 1 and g + 2 fall inside the first cue's run when it lasts that long
    at = sorted(draw(st.lists(st.one_of(st.integers(0, n), st.sampled_from([g + 1, g + 2])), min_size=1, max_size=4)))
    rows, codes, bad = [], [], []
    for p in range(n + 1):
        while at and at[0] == p:
            at.pop(0)
            eeg, code = draw(_bad_row(model, signal[min(p, n - 1)], stim[min(p, n - 1)]))
            bad.append(len(rows))
            rows.append(eeg)
            codes.append(code)
        if p < n:
            rows.append(signal[p])
            codes.append(float(stim[p]))
    return np.array(rows), np.array(codes, dtype=np.float64), tuple(bad)
