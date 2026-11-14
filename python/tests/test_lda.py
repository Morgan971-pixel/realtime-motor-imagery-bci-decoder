import numpy as np
import pytest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from mi_ref.events import LEFT, RIGHT
from mi_ref.lda import fit_lda, predict


def _features(seed=0, n=60, d=5, gap=1.0):
    rng = np.random.default_rng(seed)
    y = np.array([LEFT, RIGHT] * (n // 2))
    F = rng.normal(size=(n, d))
    F[y == RIGHT] += gap
    return F, y


def test_matches_sklearn_shrinkage_lda():
    F, y = _features()
    w, b = fit_lda(F, y, alpha=0.1)
    sk = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=0.1).fit(F, y)
    np.testing.assert_allclose(w, sk.coef_[0], rtol=1e-10)
    np.testing.assert_allclose(b, sk.intercept_[0], rtol=1e-10, atol=1e-12)


def test_unbalanced_classes_keep_equal_prior_intercept():
    F, y = _features(n=64)
    keep = np.r_[np.flatnonzero(y == LEFT)[:14], np.flatnonzero(y == RIGHT)[:18]]
    F, y = F[keep], y[keep]
    w, b = fit_lda(F, y, alpha=0.1)
    sk = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=0.1).fit(F, y)
    np.testing.assert_allclose(w, sk.coef_[0], rtol=1e-10)
    # scikit-learn adds log(prior ratio); we deliberately do not
    np.testing.assert_allclose(b, sk.intercept_[0] - np.log(18 / 14), rtol=1e-10)


def test_separable_data_is_classified_perfectly():
    F, y = _features(gap=10.0)
    w, b = fit_lda(F, y)
    np.testing.assert_array_equal(predict(w, b, F), y)


def test_score_of_exactly_zero_is_left():
    assert predict(np.array([1.0]), 0.0, np.array([[0.0]]))[0] == LEFT


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda F, y: (F[y == LEFT], y[y == LEFT]), "two LEFT and two RIGHT"),
        (lambda F, y: (F[[0, 1, 2]], y[[0, 1, 2]]), "two LEFT and two RIGHT"),
        (lambda F, y: (F, np.where(np.arange(len(y)) == 0, 0, y)), "labels must be"),
        (lambda F, y: (np.where(np.arange(F.size).reshape(F.shape) == 0, np.nan, F), y), "NaN or infinite"),
    ],
)
def test_bad_training_input_is_a_clear_error(mutate, message):
    F, y = mutate(*_features())
    with pytest.raises(ValueError, match=message):
        fit_lda(F, y)
