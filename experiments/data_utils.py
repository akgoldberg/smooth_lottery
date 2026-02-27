"""Data loading (real review matrices) and synthetic data generators."""

import os
import numpy as np
import pandas as pd

_BASE = os.path.dirname(os.path.dirname(__file__))
_BASELINES_ROOT = (
    os.path.join(_BASE, "merit_baselines")
    if os.path.isdir(os.path.join(_BASE, "merit_baselines"))
    else os.path.join(_BASE, "baselines")
)
_NEURIPS_PATH = os.path.join(
    _BASELINES_ROOT, "data", "ConferenceReviewData",
    "neurips2024_data", "neurips2024_reviews.csv",
)
_ICLR_PATH = os.path.join(
    _BASELINES_ROOT, "data", "ConferenceReviewData",
    "iclr2025_data", "iclr2025_reviews.csv",
)
_SWISS_NSF_INTERVALS_PATH = os.path.join(
    _BASELINES_ROOT, "data", "SwissNSFData", "intervals.csv",
)
_SWISS_NSF_MINT_SECTIONS_PATH = os.path.join(
    _BASELINES_ROOT, "data", "SwissNSFData", "mint_sections.csv",
)


# ---------------------------------------------------------------------------
# Real data loaders
# ---------------------------------------------------------------------------

def load_review_matrix(dataset: str, path: str = None,
                       drop_rejected: bool = False) -> tuple:
    """Load raw reviews and pivot into an (n_papers, max_reviews) matrix.

    Parameters
    ----------
    dataset : str
        One of "neurips2024" or "iclr2025".
    path : str, optional
        Override default CSV path.
    drop_rejected : bool
        If True, exclude rejected papers.

    Returns
    -------
    X : np.ndarray, shape (n, max_reviews), NaN-padded
    paper_ids : np.ndarray
    decisions : np.ndarray of str
    """
    if path is None:
        path = {"neurips2024": _NEURIPS_PATH, "iclr2025": _ICLR_PATH}[dataset]

    df = pd.read_csv(path)
    if drop_rejected:
        df = df[~df["decision"].isin(["Reject", "Withdrawn"])].reset_index(drop=True)

    # pivot: each row = paper, each column = review_number
    pivot = df.pivot_table(
        index="paper_id", columns="review_number", values="rating", aggfunc="first",
    )
    # keep only papers with >= 2 reviews
    pivot = pivot.dropna(thresh=2)

    paper_ids = pivot.index.values
    X = pivot.values.astype(float)  # NaN where review is missing

    # decisions per paper
    dec = df.drop_duplicates("paper_id").set_index("paper_id")["decision"]
    decisions = dec.reindex(paper_ids).values

    return X, paper_ids, decisions


def load_swiss_nsf_point_estimates(path: str = None) -> tuple:
    """Load Swiss NSF `mint_sections.csv` into a review matrix and row means.

    Returns
    -------
    X : np.ndarray, shape (n_proposals, max_reviews), NaN-padded
    theta : np.ndarray, shape (n_proposals,)
        Mean of observed `num_grade` values per proposal.
    """
    if path is None:
        path = _SWISS_NSF_MINT_SECTIONS_PATH

    df = pd.read_csv(path)
    grouped = (df.groupby("proposal", sort=True)["num_grade"]
                 .apply(list)
                 .reset_index())

    lengths = grouped["num_grade"].apply(len).to_numpy()
    max_reviews = int(lengths.max())
    n = len(grouped)
    X = np.full((n, max_reviews), np.nan, dtype=float)
    for i, scores in enumerate(grouped["num_grade"]):
        vals = np.asarray(scores, dtype=float)
        X[i, :len(vals)] = vals

    theta = np.nanmean(X, axis=1)
    return X, theta


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

def generate_gaussian_reviews(
    n: int, r: int,
    sigma_theta: float = 2.0,
    sigma_err: float = 1.0,
    score_range: tuple = (1, 10),
    rng: np.random.Generator = None,
) -> tuple:
    """Gaussian model: theta_i ~ N(5.5, sigma_theta), X_{ij} = theta_i + eps_{ij}.

    Returns (X, theta) where X is (n, r) clipped to score_range.
    """
    if rng is None:
        rng = np.random.default_rng()
    mid = (score_range[0] + score_range[1]) / 2
    theta = rng.normal(mid, sigma_theta, size=n)
    eps = rng.normal(0, sigma_err, size=(n, r))
    X = np.clip(theta[:, None] + eps, score_range[0], score_range[1])
    return X, theta


def generate_beta_reviews(
    n: int, r: int,
    alpha_theta: float = 2.0,
    beta_theta: float = 2.0,
    kappa: float = 20.0,
    rng: np.random.Generator = None,
) -> tuple:
    """Beta-on-[0,1] model with reviewer noise around latent quality.

    theta_i ~ Beta(alpha_theta, beta_theta)
    X_ij | theta_i ~ Beta(theta_i*kappa, (1-theta_i)*kappa)
    """
    if rng is None:
        rng = np.random.default_rng()
    theta = rng.beta(alpha_theta, beta_theta, size=n)

    # Numerical floor prevents invalid Beta params near 0/1.
    eps = 1e-6
    a = np.clip(theta * kappa, eps, None)
    b = np.clip((1.0 - theta) * kappa, eps, None)
    X = rng.beta(a[:, None], b[:, None], size=(n, r))
    return X, theta


def generate_worstcase_regret(
    n: int, k: int, r: int,
    gap: float = 1.0,
    noise: float = 1e-6,
    rng: np.random.Generator = None,
) -> tuple:
    """Two-block worst case from Theorem 2 proof.

    Top-k candidates have utility 0.5 + gap/2, bottom (n-k) have 0.5 - gap/2.
    Reviews are r copies of the true utility plus tiny noise.
    """
    if rng is None:
        rng = np.random.default_rng()
    theta = np.empty(n)
    theta[:k] = 0.5 + gap / 2
    theta[k:] = 0.5 - gap / 2
    eps = rng.normal(0, noise, size=(n, r))
    X = theta[:, None] + eps
    return X, theta


def generate_constant_reviews(n: int, r: int, value: float = 0.5) -> np.ndarray:
    """All-equal review matrix for smoothness tightness tests."""
    return np.full((n, r), value)
