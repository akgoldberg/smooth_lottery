"""Shared utilities for experiment scripts."""

from typing import Optional, Tuple

import numpy as np

from data_utils import normalize_with_review_scale, normalized_tick_size


def reviews_per_item(X: np.ndarray) -> np.ndarray:
    return np.sum(~np.isnan(X), axis=1)


def r_min(X: np.ndarray) -> int:
    counts = reviews_per_item(X)
    return int(np.min(counts[counts > 0]))


def r_med(X: np.ndarray) -> int:
    counts = reviews_per_item(X)
    return int(np.median(counts[counts > 0]))


def drop_low_review_outliers(X: np.ndarray, ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    counts = reviews_per_item(X)
    med = int(np.median(counts[counts > 0]))
    keep = counts >= max(2, med - 1)
    return X[keep], ids[keep]


def k_from_name(n: int, k_name: str) -> int:
    if k_name == "k1":
        return 1
    if k_name == "k10pct":
        return max(1, min(int(0.1 * n), n - 1))
    if k_name == "k33pct":
        return max(1, min(int(round(n / 3.0)), n - 1))
    if k_name == "k50pct":
        return max(1, min(int(0.5 * n), n - 1))
    raise ValueError(f"Unknown k_name: {k_name}")


def normalize_scores(
    X: np.ndarray,
    dataset_key: str,
    theta: Optional[np.ndarray] = None,
    synthetic_ticks: int = 10,
) -> Tuple[np.ndarray, Optional[np.ndarray], dict]:
    """Normalize using configured review scales, with synthetic tick metadata."""
    X_norm, theta_norm, meta = normalize_with_review_scale(X, dataset_key=dataset_key, theta=theta)
    meta["normalized_tick_size"] = normalized_tick_size(dataset_key, synthetic_ticks=synthetic_ticks)
    return X_norm, theta_norm, meta

