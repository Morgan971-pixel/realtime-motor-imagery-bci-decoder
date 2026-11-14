import numpy as np
import pytest
from scipy.signal import sosfilt

from mi_ref.filters import DEFAULT_BANDS, design_band, filter_causal, remove_offset, sosfilt_df2t

FS = 250.0


def _steady_amplitude(sos, freq_hz):
    t = np.arange(int(10 * FS)) / FS
    y = filter_causal(sos, np.sin(2 * np.pi * freq_hz * t)[:, None])[:, 0]
    return np.max(np.abs(y[-int(2 * FS):]))


def _realistic_channel(seed=0, n=3000):
    """Raw-EEG-like input: a huge DC offset plus noise, offset removed as the decoder does."""
    rng = np.random.default_rng(seed)
    raw = 7e5 + rng.normal(0.0, 10.0, size=(n, 1))
    return remove_offset(raw)[:, 0]


def test_band_shape_is_four_sections():
    assert design_band(8.0, 12.0, FS).shape == (4, 6)


def test_band_passes_inside_and_blocks_outside():
    sos = design_band(8.0, 12.0, FS)
    assert _steady_amplitude(sos, 10.0) > 0.9
    assert _steady_amplitude(sos, 40.0) < 0.01


@pytest.mark.parametrize("edge_hz", [8.0, 12.0])
def test_band_edges_are_minus_3_db(edge_hz):
    assert _steady_amplitude(design_band(8.0, 12.0, FS), edge_hz) == pytest.approx(1 / np.sqrt(2), abs=0.02)


def test_filter_is_causal():
    x = np.zeros((500, 1))
    x[200, 0] = 1.0
    y = filter_causal(design_band(8.0, 12.0, FS), x)
    assert np.all(y[:200] == 0.0)
    assert np.any(y[200:] != 0.0)


@pytest.mark.parametrize("band", DEFAULT_BANDS)
def test_df2t_spec_matches_scipy_on_every_band(band):
    x = _realistic_channel()
    sos = design_band(*band, FS)
    np.testing.assert_allclose(sosfilt_df2t(sos, x), filter_causal(sos, x[:, None])[:, 0], rtol=0, atol=1e-12)


def test_df2t_spec_rejects_multichannel_input():
    with pytest.raises(ValueError, match="one channel"):
        sosfilt_df2t(design_band(8.0, 12.0, FS), np.zeros((10, 2)))


def test_chunked_filtering_with_carried_state_equals_one_pass():
    """The property streaming relies on: filter state carried across chunks changes nothing."""
    x = _realistic_channel(seed=1)
    sos = design_band(8.0, 12.0, FS)
    first, state = sosfilt(sos, x[:777], zi=np.zeros((sos.shape[0], 2)))
    second, _ = sosfilt(sos, x[777:], zi=state)
    np.testing.assert_array_equal(np.concatenate([first, second]), sosfilt(sos, x))


def test_remove_offset_zeroes_first_sample():
    x = np.array([[700000.0, 5.0], [700001.0, 6.0]])
    np.testing.assert_array_equal(remove_offset(x), [[0.0, 0.0], [1.0, 1.0]])
