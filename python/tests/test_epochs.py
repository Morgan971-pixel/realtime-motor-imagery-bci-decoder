import pytest

from mi_ref.epochs import decision_sample, to_samples, window_slice


def test_window_is_half_second_to_two_and_a_half_seconds():
    assert window_slice(1000, 250.0) == slice(1125, 1625)


def test_decision_happens_on_last_window_sample():
    assert decision_sample(1000, 250.0) == 1624


def test_half_samples_round_away_from_zero_like_rust_and_cpp():
    # 0.5 s * 125 Hz = 62.5 samples: Python's round() would give 62, Rust/C++ give 63
    assert to_samples(0.5, 125.0) == 63
    assert window_slice(0, 125.0) == slice(63, 313)


@pytest.mark.parametrize("fs", [100.0, 125.0, 128.0, 250.0, 256.0, 500.0])
def test_window_length_matches_to_samples(fs):
    sl = window_slice(10, fs, start_s=0.25, len_s=1.5)
    assert sl.stop - sl.start == to_samples(1.5, fs)
    assert decision_sample(10, fs, 0.25, 1.5) == sl.stop - 1


def test_negative_durations_are_rejected():
    with pytest.raises(ValueError):
        to_samples(-0.5, 250.0)
