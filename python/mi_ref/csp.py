"""Common Spatial Patterns: channel mixes whose variance best separates left from right.

W rows 0..n_pairs-1 are RIGHT-dominant filters (smallest eigenvalues), the last n_pairs
rows are LEFT-dominant (largest). Features use centered population variance (ddof=0),
floored at VAR_FLOOR before the log, in training and in every native core (fmax: a NaN
variance also takes the floor, as in Rust and C++).
"""
import numpy as np
from scipy.linalg import eigh

from .events import LEFT, RIGHT

VAR_FLOOR = 1e-12  # a flat or all-zero window gives log(1e-12), never -inf


def shrink(cov: np.ndarray, alpha: float) -> np.ndarray:
    """Blend toward a scaled identity: (1 - alpha) * C + alpha * (trace(C) / n) * I."""
    n = cov.shape[0]
    return (1.0 - alpha) * cov + alpha * (np.trace(cov) / n) * np.eye(n)


def normalized_cov(x: np.ndarray) -> np.ndarray:
    """x is (channels, time). Centered, then trace-normalized so loud trials do not dominate."""
    x = x - x.mean(axis=1, keepdims=True)
    c = x @ x.T
    trace = np.trace(c)
    if trace <= 0.0:
        raise ValueError("trial has zero variance on every channel")
    return c / trace


def fit_csp(X: np.ndarray, y: np.ndarray, n_pairs: int = 2, alpha: float = 0.1) -> np.ndarray:
    """X is (trials, channels, time). Returns W of shape (2 * n_pairs, channels)."""
    y = np.asarray(y)
    if 2 * n_pairs > X.shape[1]:
        raise ValueError(f"2 * n_pairs ({2 * n_pairs}) exceeds the channel count ({X.shape[1]})")
    if not (np.any(y == LEFT) and np.any(y == RIGHT)):
        raise ValueError("need at least one LEFT and one RIGHT trial")
    c_left = shrink(np.mean([normalized_cov(x) for x in X[y == LEFT]], axis=0), alpha)
    c_right = shrink(np.mean([normalized_cov(x) for x in X[y == RIGHT]], axis=0), alpha)
    _, vecs = eigh(c_left, c_left + c_right)  # eigenvalues ascending
    pick = np.r_[np.arange(n_pairs), np.arange(vecs.shape[1] - n_pairs, vecs.shape[1])]
    return vecs[:, pick].T


def log_variance(W: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Features for one window x (channels, time): log of each filtered signal's variance."""
    return np.log(np.fmax(np.var(W @ x, axis=1), VAR_FLOOR))


def log_variance_batch(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    """X is (trials, channels, time) -> (trials, 2 * n_pairs)."""
    return np.log(np.fmax(np.var(np.einsum("kc,nct->nkt", W, X), axis=2), VAR_FLOOR))
