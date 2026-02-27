"""Unified mechanism interface: review matrix X -> probability vector p.

Every public mechanism function has signature:
    mechanism(X: ndarray[n, r], k: int, **params) -> ndarray[n]
"""

import sys
import os
import numpy as np

_BASE = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, _BASE)
sys.path.insert(0, os.path.join(_BASE, "merit_baselines"))

from smooth_lotteries import linear_lottery_smooth, softmax_topk_smooth
from algorithm.merit import run_merit
from algorithm.helpers import swiss_nsf as _swiss_nsf_intervals


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nanmean(X: np.ndarray) -> np.ndarray:
    """Mean review score per candidate, ignoring NaN."""
    return np.nanmean(X, axis=1)


def _effective_r(X: np.ndarray) -> int:
    """Minimum number of non-NaN reviews per candidate (for worst-case bounds)."""
    counts = np.sum(~np.isnan(X), axis=1)
    counts = counts[counts > 0]
    return int(np.min(counts))


def reviews_to_intervals(X: np.ndarray, method: str = "leave_one_out") -> tuple:
    """Convert review matrix to (intervals, point_estimates).

    Parameters
    ----------
    X : ndarray, shape (n, r), may contain NaN
    method : str
        "leave_one_out" - interval = (min, max) of leave-one-out means
        "gaussian_ci"   - point_est +/- 1.96 * std / sqrt(r)
        "minmax"        - (min review, max review)

    Returns
    -------
    intervals : list of (lo, hi) tuples
    x : ndarray of point estimates
    """
    n = X.shape[0]
    x = _nanmean(X)

    if method == "leave_one_out":
        intervals = []
        for i in range(n):
            row = X[i, ~np.isnan(X[i])]
            if len(row) <= 1:
                intervals.append((float(row.mean()), float(row.mean())))
                continue
            loo_means = []
            for j in range(len(row)):
                loo = np.delete(row, j).mean()
                loo_means.append(loo)
            intervals.append((min(loo_means), max(loo_means)))
    elif method == "gaussian_ci":
        intervals = []
        for i in range(n):
            row = X[i, ~np.isnan(X[i])]
            se = row.std(ddof=1) / np.sqrt(len(row)) if len(row) > 1 else 0.0
            intervals.append((float(x[i] - 1.96 * se), float(x[i] + 1.96 * se)))
    elif method == "minmax":
        intervals = []
        for i in range(n):
            row = X[i, ~np.isnan(X[i])]
            intervals.append((float(row.min()), float(row.max())))
    else:
        raise ValueError(f"Unknown interval method: {method}")

    return intervals, x


# ---------------------------------------------------------------------------
# Mechanisms
# ---------------------------------------------------------------------------

def linear_lottery_mechanism(X: np.ndarray, k: int, L: float) -> np.ndarray:
    """Linear Lottery with L-smoothness guarantee."""
    v = _nanmean(X)
    r = _effective_r(X)
    D_v = 1.0 / r
    p, _ = linear_lottery_smooth(v, k, L, D_v)
    return p


def softmax_mechanism(X: np.ndarray, k: int, L: float,
                      n_samples: int = 50_000,
                      rng: np.random.Generator = None) -> np.ndarray:
    """Top-k Softmax (Gumbel-top-k) with L-smoothness guarantee."""
    v = _nanmean(X)
    r = _effective_r(X)
    D_v = 1.0 / r
    p, _ = softmax_topk_smooth(v, k, L, D_v, n_samples=n_samples, rng=rng)
    return p


def merit_mechanism(X: np.ndarray, k: int,
                    interval_method: str = "leave_one_out") -> np.ndarray:
    """MERIT mechanism: X -> intervals -> run_merit."""
    intervals, _ = reviews_to_intervals(X, method=interval_method)
    p, _ = run_merit(intervals, k)
    return np.asarray(p)


def swiss_nsf_mechanism(X: np.ndarray, k: int,
                        interval_method: str = "leave_one_out") -> np.ndarray:
    """Swiss NSF procedure: X -> intervals -> swiss_nsf."""
    intervals, x = reviews_to_intervals(X, method=interval_method)
    p = _swiss_nsf_intervals(intervals, list(x), k)
    return np.asarray(p, dtype=float)


def randomize_above_threshold(X: np.ndarray, k: int, m: int) -> np.ndarray:
    """Rank-based randomize-above-threshold.

    Accept top (k - m) deterministically, uniform lottery among ranks
    k-m+1 to k+m. Parameter m controls the band half-width.
    """
    v = _nanmean(X)
    n = len(v)
    m = min(m, k, n - k)  # clamp
    ranking = np.argsort(-v)  # descending

    p = np.zeros(n)
    # top (k - m) are auto-accepted
    for idx in ranking[:k - m]:
        p[idx] = 1.0
    # next 2m are in the lottery pool
    pool = ranking[k - m: k + m]
    if len(pool) > 0:
        prob = float(m) / len(pool)  # need m more acceptances from 2m candidates
        for idx in pool:
            p[idx] = prob
    return p


def top_k_mechanism(X: np.ndarray, k: int) -> np.ndarray:
    """Deterministic top-k by mean review score."""
    v = _nanmean(X)
    p = np.zeros(len(v))
    top = np.argsort(-v)[:k]
    p[top] = 1.0
    return p
