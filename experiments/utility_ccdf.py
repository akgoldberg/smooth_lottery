"""CCDF of normalized utilities for main datasets."""

import argparse
import os
import sys

import numpy as np
import pandas as pd

_BASE = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, _BASE)

from data_utils import (
    generate_beta_reviews,
    load_review_matrix,
    load_swiss_nsf_point_estimates,
)
from utils import drop_low_review_outliers, normalize_scores
from plot_results import plot_utility_ccdf_curves


DATASETS = ["beta", "iclr", "neurips", "swissnsf"]
LABELS = {
    "beta": "Beta",
    "iclr": "ICLR",
    "neurips": "NeurIPS",
    "swissnsf": "Swiss NSF",
}
COLORS = {
    "beta": "#2ca02c",
    "iclr": "#1f77b4",
    "neurips": "#ff7f0e",
    "swissnsf": "#d62728",
}


def load_utilities(name: str, args, rng: np.random.Generator) -> np.ndarray:
    if name == "iclr":
        X, ids, _ = load_review_matrix("iclr2025", drop_rejected=True)
        X, _ = drop_low_review_outliers(X, ids)
        Xn, _, _ = normalize_scores(X, dataset_key="iclr2025", synthetic_ticks=10)
        return np.nanmean(Xn, axis=1)
    if name == "neurips":
        X, ids, _ = load_review_matrix("neurips2024", drop_rejected=True)
        X, _ = drop_low_review_outliers(X, ids)
        Xn, _, _ = normalize_scores(X, dataset_key="neurips2024", synthetic_ticks=10)
        return np.nanmean(Xn, axis=1)
    if name == "swissnsf":
        X, theta = load_swiss_nsf_point_estimates()
        _, theta_n, _ = normalize_scores(X, dataset_key="swissnsf", theta=theta, synthetic_ticks=10)
        return theta_n
    if name == "beta":
        X, theta = generate_beta_reviews(
            n=args.beta_n,
            r=args.beta_r,
            alpha_theta=args.beta_alpha,
            beta_theta=args.beta_beta,
            kappa=args.beta_kappa,
            rng=rng,
        )
        _, theta_n, _ = normalize_scores(X, dataset_key="synthetic", theta=theta, synthetic_ticks=10)
        return theta_n
    raise ValueError(name)


def ccdf(values: np.ndarray):
    v = np.sort(np.asarray(values, dtype=float))
    n = len(v)
    y = 1.0 - (np.arange(1, n + 1) / n)
    return v, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Utility CCDF across datasets")
    parser.add_argument("--beta_n", type=int, default=200)
    parser.add_argument("--beta_r", type=int, default=5)
    parser.add_argument("--beta_alpha", type=float, default=2.0)
    parser.add_argument("--beta_beta", type=float, default=2.0)
    parser.add_argument("--beta_kappa", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default=os.path.join(os.path.dirname(__file__), "results"))
    parser.add_argument("--fig_dir", default=os.path.join(os.path.dirname(__file__), "figures"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    curves = []
    rows = []
    for ds in DATASETS:
        u = load_utilities(ds, args, rng)
        x, y = ccdf(u)
        curves.append({"x": x, "y": y, "label": LABELS[ds], "color": COLORS[ds]})
        rows.append(
            {
                "dataset": ds,
                "n": int(len(u)),
                "mean_u": float(np.mean(u)),
                "std_u": float(np.std(u)),
                "u_p90": float(np.quantile(u, 0.90)),
                "u_p75": float(np.quantile(u, 0.75)),
                "u_p50": float(np.quantile(u, 0.50)),
                "u_p25": float(np.quantile(u, 0.25)),
                "u_p10": float(np.quantile(u, 0.10)),
                "gap_p90_p80": float(np.quantile(u, 0.90) - np.quantile(u, 0.80)),
                "gap_p80_p70": float(np.quantile(u, 0.80) - np.quantile(u, 0.70)),
            }
        )

    out_pdf = os.path.join(args.fig_dir, "utility_ccdf_datasets.pdf")
    out_csv = os.path.join(args.output_dir, "utility_ccdf_summary.csv")
    plot_utility_ccdf_curves(curves, out_pdf=out_pdf)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved {out_pdf}")
    print(f"Saved {out_csv}")
