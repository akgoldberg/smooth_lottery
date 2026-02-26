"""
Smooth selection rules from "Smooth Partial Lotteries".

L-smoothness guarantee (Definition 1):
  ||p(X) - p(X')||_1 <= L * ||X - X'||_{1,1}

Parameter choices:
  Linear Lottery : w   = L / (2 * D_v)   [Theorem 1]
  Top-k Softmax  : tau = 2 * D_v / (e*L) [Theorem 4]

All mechanism functions return (p, sample).
For softmax p is estimated via Monte Carlo, for linear lottery p is exact.
"""

import math
import numpy as np
from typing import Optional


def mean_utility(X: np.ndarray) -> np.ndarray:
    """Mean review score per candidate. D_v = 1/r."""
    return X.mean(axis=1)


def systematic_sampling(k: int, p: np.ndarray) -> list:
    """
    Sample exactly k items with marginal probabilities p (summing to k).

    Source: Madow & Madow (1944), "On the Theory of Systematic Sampling II"
    https://projecteuclid.org/journals/annals-of-mathematical-statistics/
    volume-20/issue-3/On-the-Theory-of-Systematic-Sampling-II/
    10.1214/aoms/1177729988.full
    """
    n = len(p)
    assert np.isclose(sum(p), k), "Marginal probabilities must sum to k"

    perm = np.random.permutation(n)
    p_perm = [p[i] for i in perm]

    S = np.cumsum(p_perm)
    S = np.insert(S, 0, 0)

    u = np.random.uniform(0, 1)
    sampling_points = [u + m for m in range(k)]

    selected = []
    j = 0
    for point in sampling_points:
        while j < len(S) and S[j] <= point:
            j += 1
        selected.append(perm[j - 1])

    return selected


def linear_lottery(v: np.ndarray, k: int, w: float) -> tuple:
    """
    Linear Lottery: p_i = clip[0,1](w * v_i + b), with b chosen so sum(p) = k.

    Equivalently, the projection of w*v onto C_{n,k} = {p in [0,1]^n : sum(p) = k}
    (Proposition 1). Intercept found via breakpoint search (Algorithm 1).

    Returns (p, sample) where sample is drawn via systematic sampling.
    """
    if w <= 0:
        raise ValueError("slope w must be positive")
    n = len(v)
    if not (0 < k < n):
        raise ValueError(f"k must satisfy 0 < k < n, got k={k}, n={n}")

    z = w * v
    # Breakpoints of S(b) = sum_i clip[0,1](z_i + b) are at b = -z_i and b = 1 - z_i
    breakpoints = np.sort(np.concatenate([-z, 1.0 - z]))

    def S(b: float) -> float:
        return float(np.clip(z + b, 0.0, 1.0).sum())

    b = None
    for j in range(len(breakpoints) - 1):
        b_lo, b_hi = breakpoints[j], breakpoints[j + 1]
        S_lo, S_hi = S(b_lo), S(b_hi)
        if S_lo <= k <= S_hi:
            b = b_lo if S_lo == S_hi else b_lo + (b_hi - b_lo) * (k - S_lo) / (S_hi - S_lo)
            break

    if b is None:
        b = breakpoints[0] if k <= S(breakpoints[0]) else breakpoints[-1]

    p = np.clip(z + b, 0.0, 1.0)
    p = p / p.sum() * k  # enforce budget exactly
    return p, systematic_sampling(k, p)


def linear_lottery_smooth(v: np.ndarray, k: int, L: float, D_v: float) -> tuple:
    """
    Linear Lottery with w = L / (2 * D_v), guaranteeing L-smoothness (Theorem 1).
    D_v = 1/r for mean utility with r reviews per candidate.

    Returns (p, sample).
    """
    if L <= 0:
        raise ValueError("L must be positive")
    if D_v <= 0:
        raise ValueError("D_v must be positive")
    return linear_lottery(v, k, w=L / (2.0 * D_v))


def softmax_topk(
    v: np.ndarray,
    k: int,
    tau: float,
    n_samples: int = 20_000,
    rng: Optional[np.random.Generator] = None,
) -> tuple:
    """
    Top-k Softmax (Gumbel-top-k): estimates marginals via Monte Carlo,
    then draws one sample via systematic sampling.

    p_i estimated as fraction of Gumbel-top-k draws containing item i,
    where each draw adds i.i.d. Gumbel(0, tau) noise to v and takes the top k.

    Smoothness guarantee (Theorem 4): L <= 2 * D_v / (e * tau)

    Returns (p, sample).
    """
    if tau <= 0:
        raise ValueError("temperature tau must be positive")
    n = len(v)
    if not (0 < k < n):
        raise ValueError(f"k must satisfy 0 < k < n, got k={k}, n={n}")

    if rng is None:
        rng = np.random.default_rng()

    # Gumbel(0, tau) via inverse CDF: -tau * log(-log(U))
    U = rng.uniform(size=(n_samples, n))
    gumbel = -tau * np.log(-np.log(np.clip(U, 1e-10, 1.0 - 1e-10)))

    topk_idx = np.argpartition(v[np.newaxis, :] + gumbel, -k, axis=1)[:, -k:]
    counts = np.zeros(n)
    for row in topk_idx:
        counts[row] += 1

    p = counts / n_samples
    p = p / p.sum() * k  # enforce budget exactly
    return p, systematic_sampling(k, p)


def softmax_topk_smooth(
    v: np.ndarray,
    k: int,
    L: float,
    D_v: float,
    n_samples: int = 20_000,
    rng: Optional[np.random.Generator] = None,
) -> tuple:
    """
    Top-k Softmax with tau = 2 * D_v / (e * L), guaranteeing L-smoothness (Theorem 4).

    Returns (p, sample).
    """
    if L <= 0:
        raise ValueError("L must be positive")
    if D_v <= 0:
        raise ValueError("D_v must be positive")
    return softmax_topk(v, k, tau=2.0 * D_v / (math.e * L), n_samples=n_samples, rng=rng)


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    n, r, k = 10, 3, 3
    X = rng.uniform(size=(n, r))
    v = mean_utility(X)
    D_v = 1.0 / r
    L = 2.0

    print("v:", np.round(v, 3))
    print()

    p, sample = linear_lottery(v, k, w=L / (2 * D_v))
    print(f"Linear Lottery (w={L/(2*D_v):.2f}):  p={np.round(p,3)}  sample={sorted(sample)}")

    p, sample = linear_lottery_smooth(v, k, L=L, D_v=D_v)
    print(f"Linear Lottery (L={L}, D_v={D_v:.4f}): p={np.round(p,3)}  sample={sorted(sample)}")
    print()

    tau = 2 * D_v / (math.e * L)
    p, sample = softmax_topk(v, k, tau=tau, n_samples=50_000, rng=rng)
    print(f"Softmax (tau={tau:.4f}):          p={np.round(p,3)}  sample={sorted(sample)}")

    p, sample = softmax_topk_smooth(v, k, L=L, D_v=D_v, n_samples=50_000, rng=rng)
    print(f"Softmax (L={L}, D_v={D_v:.4f}):      p={np.round(p,3)}  sample={sorted(sample)}")