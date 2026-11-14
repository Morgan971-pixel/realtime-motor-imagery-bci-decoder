"""Shrinkage LDA in closed form. score = w . f + b; score > 0 means RIGHT, otherwise LEFT.

w matches scikit-learn's LinearDiscriminantAnalysis(solver="lsqr", shrinkage=alpha) for any
class balance. b assumes EQUAL priors on purpose: in a cued left/right task the live class
ratio is 50/50, and a training fold's imbalance should not bias the decoder. So b equals
scikit-learn's intercept only when both classes have the same number of trials.
A score of exactly 0 is LEFT (strict >), in every implementation.
"""
import numpy as np

from .csp import shrink
from .events import LEFT, RIGHT


def fit_lda(F: np.ndarray, y: np.ndarray, alpha: float = 0.1) -> tuple[np.ndarray, float]:
    """F is (trials, features). Pooled covariance divides by n (matches scikit-learn)."""
    F, y = np.asarray(F, dtype=np.float64), np.asarray(y)
    if not set(np.unique(y)) <= {LEFT, RIGHT}:
        raise ValueError(f"labels must be {LEFT} (left) or {RIGHT} (right), got {sorted(set(np.unique(y)))}")
    if min(np.sum(y == LEFT), np.sum(y == RIGHT)) < 2:
        raise ValueError("need at least two LEFT and two RIGHT trials")
    if not np.all(np.isfinite(F)):
        raise ValueError("features contain NaN or infinite values")
    f_left, f_right = F[y == LEFT], F[y == RIGHT]
    mu_left, mu_right = f_left.mean(axis=0), f_right.mean(axis=0)
    scatter = (f_left - mu_left).T @ (f_left - mu_left) + (f_right - mu_right).T @ (f_right - mu_right)
    cov = shrink(scatter / len(F), alpha)
    w = np.linalg.solve(cov, mu_right - mu_left)
    b = -float(w @ (mu_left + mu_right)) / 2.0
    return w, b


def predict(w: np.ndarray, b: float, F: np.ndarray) -> np.ndarray:
    """Labels for rows of F: RIGHT where w . f + b > 0, else LEFT."""
    return np.where(F @ w + b > 0.0, RIGHT, LEFT)
