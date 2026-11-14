import numpy as np
import pytest

from mi_ref.events import LEFT, RIGHT, imagery_markers, markers_from_stim
from mi_ref.filters import design_band, filter_causal
from mi_ref.synth import C3, C4, IMAGERY_S, make_recording


def test_synthetic_protocol_shape_and_labels():
    rec = make_recording(n_trials=4, seed=0)
    assert rec.signal.shape == (len(rec.stim), 8)
    imag = imagery_markers(markers_from_stim(rec.stim))
    assert [m.movement for m in imag] == [1, 2, 1, 2]
    assert [m.trial for m in imag] == [1, 2, 3, 4]


def test_synthetic_is_deterministic_per_seed():
    a, b = make_recording(n_trials=2, seed=5), make_recording(n_trials=2, seed=5)
    np.testing.assert_array_equal(a.signal, b.signal)
    assert not np.array_equal(a.signal, make_recording(n_trials=2, seed=6).signal)


def _imagery_mu_power(effect, channel, movement):
    """9-11 Hz power during imagery on one channel, pooled over one class's trials."""
    rec = make_recording(n_trials=20, seed=0, effect=effect)
    mu = filter_causal(design_band(9.0, 11.0, 250.0), rec.signal - rec.signal[0])[:, channel] ** 2
    imag = int(IMAGERY_S * 250)
    onsets = [m.sample for m in imagery_markers(markers_from_stim(rec.stim)) if m.movement == movement]
    return sum(mu[i + 250 : i + imag].sum() for i in onsets)


@pytest.mark.parametrize(
    "movement, suppressed, untouched",
    [(LEFT, C4, C3), (RIGHT, C3, C4)],
)
def test_imagery_suppresses_mu_on_the_opposite_side_only(movement, suppressed, untouched):
    """Same seed with and without the effect: the random draws cancel, leaving the planted ERD."""
    def ratio(channel):
        return _imagery_mu_power(0.6, channel, movement) / _imagery_mu_power(0.0, channel, movement)

    assert ratio(suppressed) < 0.3  # amplitude x0.4 -> power about x0.16, plus the noise floor
    assert ratio(untouched) == pytest.approx(1.0, abs=1e-6)  # only filter ringing from neighbors
