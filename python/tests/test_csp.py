import numpy as np
import pytest

from mi_ref.csp import VAR_FLOOR, fit_csp, log_variance, log_variance_batch, shrink
from mi_ref.events import LEFT, RIGHT


def _two_class_trials(seed=0, n=40, ch=4, t=200):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, ch, t))
    y = np.array([LEFT, RIGHT] * (n // 2))
    X[y == LEFT, 0] *= 3.0  # left trials: loud channel 0
    X[y == RIGHT, 1] *= 3.0  # right trials: loud channel 1
    return X, y


def test_shrink_endpoints():
    c = np.array([[2.0, 1.0], [1.0, 4.0]])
    np.testing.assert_allclose(shrink(c, 0.0), c)
    np.testing.assert_allclose(shrink(c, 1.0), 3.0 * np.eye(2))


def test_csp_shape():
    X, y = _two_class_trials()
    assert fit_csp(X, y, n_pairs=2).shape == (4, 4)


def test_csp_features_separate_classes():
    X, y = _two_class_trials()
    W = fit_csp(X, y, n_pairs=1)
    F = log_variance_batch(W, X)
    # first filter is largest for RIGHT, last for LEFT (eigenvalues ascending)
    assert F[y == RIGHT, 0].mean() > F[y == LEFT, 0].mean() + 1.0
    assert F[y == LEFT, 1].mean() > F[y == RIGHT, 1].mean() + 1.0


def test_single_and_batch_features_agree():
    X, y = _two_class_trials()
    W = fit_csp(X, y)
    np.testing.assert_allclose(log_variance(W, X[3]), log_variance_batch(W, X)[3], rtol=1e-12)


def test_features_match_explicit_formula():
    """Pins what native ports must compute: centered, divide by N (ddof=0), natural log."""
    rng = np.random.default_rng(7)
    x = rng.normal(size=(8, 500)) + 5.0  # nonzero mean on purpose
    W = rng.normal(size=(4, 8))
    z = W @ x
    expected = np.log(np.sum((z - z.mean(axis=1, keepdims=True)) ** 2, axis=1) / 500)
    np.testing.assert_allclose(log_variance(W, x), expected, rtol=1e-12)


def test_features_ignore_dc_offset():
    X, y = _two_class_trials()
    W = fit_csp(X, y)
    np.testing.assert_allclose(log_variance(W, X[0] + 123.0), log_variance(W, X[0]), rtol=1e-9)


def test_training_ignores_dc_offset():
    X, y = _two_class_trials()
    np.testing.assert_allclose(np.abs(fit_csp(X + 5.0, y)), np.abs(fit_csp(X, y)), rtol=1e-9, atol=1e-12)


def test_flat_window_is_floored_not_minus_infinity():
    assert np.all(log_variance(np.eye(4)[:2], np.zeros((4, 50))) == np.log(VAR_FLOOR))


def test_missing_class_is_a_clear_error():
    X, y = _two_class_trials()
    with pytest.raises(ValueError, match="at least one LEFT and one RIGHT"):
        fit_csp(X[y == LEFT], y[y == LEFT])


def test_too_many_pairs_is_a_clear_error():
    X, y = _two_class_trials()
    with pytest.raises(ValueError, match="exceeds the channel count"):
        fit_csp(X, y, n_pairs=3)
