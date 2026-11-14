import numpy as np
import pytest

from mi_ref.events import LEFT, RIGHT, Marker, imagery_markers, markers_from_stim, parse_code


def test_parse_code_splits_trial_phase_movement():
    assert parse_code(111) == (1, 1, 1)
    assert parse_code(2131) == (21, 3, 1)
    assert parse_code(652) == (6, 5, 2)


@pytest.mark.parametrize("code", [0, 5, 31, 99])
def test_parse_code_rejects_trial_zero(code):
    with pytest.raises(ValueError, match="trial >= 1"):
        parse_code(code)


def test_markers_from_stim_merges_repeated_pulse():
    stim = np.array([0, 121, 0, 0, 131, 131, 0, 151], dtype=float)
    assert markers_from_stim(stim) == [
        Marker(1, 1, 2, 1),
        Marker(4, 1, 3, 1),
        Marker(7, 1, 5, 1),
    ]


def test_run_of_three_counts_once_and_reports_onset():
    assert markers_from_stim(np.array([0, 131, 131, 131, 0.0])) == [Marker(1, 1, 3, 1)]


def test_different_codes_on_adjacent_samples_are_two_markers():
    assert [m.sample for m in markers_from_stim(np.array([131, 132, 0.0]))] == [0, 1]


def test_same_code_after_a_gap_is_a_new_marker():
    assert [m.sample for m in markers_from_stim(np.array([131, 0, 131.0]))] == [0, 2]


def test_empty_and_all_zero_give_no_markers():
    assert markers_from_stim(np.array([])) == []
    assert markers_from_stim(np.zeros(5)) == []


def test_imagery_markers_keeps_only_phase_3():
    stim = np.array([121, 0, 131, 0, 141, 0, 151, 0, 232, 0], dtype=float)
    kept = imagery_markers(markers_from_stim(stim))
    assert [(m.sample, m.movement) for m in kept] == [(2, LEFT), (8, RIGHT)]


@pytest.mark.parametrize(
    "stim, message",
    [
        ([0.0, np.nan], "NaN or infinite"),
        ([0.0, np.inf], "NaN or infinite"),
        ([0.0, -131.0], "negative"),
        ([0.0, 1e30], "above"),
        ([0.0, 131.4], "non-integer"),
        ([[0.0, 131.0]], "1-D"),
        ([0.0, 31.0], "trial >= 1"),
    ],
)
def test_invalid_stim_is_an_error(stim, message):
    with pytest.raises(ValueError, match=message):
        markers_from_stim(np.array(stim))
