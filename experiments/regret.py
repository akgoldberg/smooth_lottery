"""Cross-dataset regret experiments for smooth selection rules.

Current setup only:
- Datasets: Beta, ICLR 2025, NeurIPS 2024, Swiss NSF
- k configs: 1, 10%, 50%
- L grid: linear from 0.5*(1/r_min) to 1.0
- Beta synthetic: repeated draws with error bars
- No upper/lower bound overlays in plots
"""

import argparse
import os
import sys
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd

_BASE = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, _BASE)

from mechanisms import linear_lottery_mechanism, softmax_mechanism
from data_utils import load_review_matrix, load_swiss_nsf_point_estimates, generate_beta_reviews
from plot_results import plot_regret_vs_param_sidebyside


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def compute_regret(v: np.ndarray, p: np.ndarray, k: int) -> float:
    top_k_vals = np.sort(v)[-k:]
    return float(top_k_vals.sum() - np.dot(v, p))


def reviews_per_item(X: np.ndarray) -> np.ndarray:
    return np.sum(~np.isnan(X), axis=1)


def r_min(X: np.ndarray) -> int:
    counts = reviews_per_item(X)
    return int(np.min(counts[counts > 0]))


def r_med(X: np.ndarray) -> int:
    counts = reviews_per_item(X)
    return int(np.median(counts[counts > 0]))


def compact_label(label: str) -> str:
    return (label
            .replace("Beta(2,2), kappa=20", "Beta")
            .replace("Swiss NSF (mint_sections means)", "Swiss NSF")
            .replace("NeurIPS 2024", "NeurIPS")
            .replace("ICLR 2025", "ICLR")
            .replace("Beta", "Beta"))


def normalize_to_unit_interval(
    X: np.ndarray, theta: Optional[np.ndarray]
) -> Tuple[np.ndarray, Optional[np.ndarray], dict]:
    """Affine-normalize observed review scores to [0, 1].

    Uses global min/max over non-NaN entries of X for each dataset realization.
    Applies the same transform to theta when provided, then clips theta to [0, 1].
    """
    mask = ~np.isnan(X)
    x_min = float(np.min(X[mask]))
    x_max = float(np.max(X[mask]))
    span = x_max - x_min

    if span <= 0:
        X_norm = np.zeros_like(X, dtype=float)
        theta_norm = None if theta is None else np.zeros_like(theta, dtype=float)
    else:
        X_norm = (X - x_min) / span
        theta_norm = None
        if theta is not None:
            theta_norm = np.clip((theta - x_min) / span, 0.0, 1.0)

    meta = {"score_min_raw": x_min, "score_max_raw": x_max}
    return X_norm, theta_norm, meta


def k_configs() -> List[Tuple[str, Callable[[int], int]]]:
    return [
        ("k1", lambda n: 1),
        ("k10pct", lambda n: max(1, min(int(0.1 * n), n - 1))),
        ("k50pct", lambda n: max(1, min(int(0.5 * n), n - 1))),
    ]


def run_regret_vs_L(
    X: np.ndarray,
    k: int,
    L_values: np.ndarray,
    theta: Optional[np.ndarray],
    n_softmax_samples: int,
    rng: np.random.Generator,
) -> dict:
    v = np.nanmean(X, axis=1) if theta is None else theta

    out = {"L": [], "regret_linear": [], "regret_softmax": []}
    for L in L_values:
        p_lin = linear_lottery_mechanism(X, k, L)
        p_soft = softmax_mechanism(X, k, L, n_samples=n_softmax_samples, rng=rng)
        out["L"].append(float(L))
        out["regret_linear"].append(compute_regret(v, p_lin, k))
        out["regret_softmax"].append(compute_regret(v, p_soft, k))
    return out


def aggregate_trials(results_list: List[dict]) -> dict:
    if not results_list:
        raise ValueError("results_list must be non-empty")
    x = np.asarray(results_list[0]["L"], dtype=float)

    lin = np.array([r["regret_linear"] for r in results_list], dtype=float)
    soft = np.array([r["regret_softmax"] for r in results_list], dtype=float)

    out = {
        "L": x.tolist(),
        "regret_linear": lin.mean(axis=0).tolist(),
        "regret_softmax": soft.mean(axis=0).tolist(),
        "n_trials": [len(results_list)] * len(x),
    }
    if len(results_list) > 1:
        # Plot uncertainty as standard deviation across draws.
        sd_lin = lin.std(axis=0)
        sd_soft = soft.std(axis=0)
        out["regret_linear_std"] = sd_lin.tolist()
        out["regret_softmax_std"] = sd_soft.tolist()
        out["regret_linear_sd"] = sd_lin.tolist()
        out["regret_softmax_sd"] = sd_soft.tolist()
    return out


# ---------------------------------------------------------------------------
# Data loading + filtering
# ---------------------------------------------------------------------------

def drop_low_review_outliers(
    X: np.ndarray,
    theta: Optional[np.ndarray],
    ids: np.ndarray,
    dataset: str,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, dict, List[dict]]:
    """Drop points well below r_med on real conference datasets.

    Rule: keep papers with review_count >= (r_med - 1). This removes very low-count
    outliers (e.g., 2-review papers when r_med=4) while preserving the bulk.
    """
    counts = reviews_per_item(X)
    med = int(np.median(counts[counts > 0]))
    min_keep = max(2, med - 1)

    keep = counts >= min_keep
    dropped_idx = np.where(~keep)[0]

    summary = {
        "dataset": dataset,
        "n_before": int(X.shape[0]),
        "n_after": int(np.sum(keep)),
        "r_med_before": med,
        "min_keep_reviews": int(min_keep),
        "dropped_count": int(len(dropped_idx)),
    }

    dropped_rows = []
    for i in dropped_idx:
        dropped_rows.append({
            "dataset": dataset,
            "paper_id": str(ids[i]),
            "reviews": int(counts[i]),
            "r_med_before": med,
            "min_keep_reviews": int(min_keep),
        })

    X2 = X[keep]
    theta2 = theta[keep] if theta is not None else None
    ids2 = ids[keep]
    return X2, theta2, ids2, summary, dropped_rows


def load_dataset(data_name: str, args, rng):
    """Return X, theta, label, ids, drop_summary, dropped_rows."""
    if data_name == "neurips":
        X, paper_ids, _ = load_review_matrix("neurips2024", drop_rejected=True)
        X, theta, ids, summary, rows = drop_low_review_outliers(
            X, None, paper_ids, dataset="neurips"
        )
        X, theta, norm_meta = normalize_to_unit_interval(X, theta)
        summary.update(norm_meta)
        return X, theta, "NeurIPS 2024", ids, summary, rows

    if data_name == "iclr":
        X, paper_ids, _ = load_review_matrix("iclr2025", drop_rejected=True)
        X, theta, ids, summary, rows = drop_low_review_outliers(
            X, None, paper_ids, dataset="iclr"
        )
        X, theta, norm_meta = normalize_to_unit_interval(X, theta)
        summary.update(norm_meta)
        return X, theta, "ICLR 2025", ids, summary, rows

    if data_name == "swissnsf":
        X, theta = load_swiss_nsf_point_estimates()
        ids = np.array([f"swiss_{i}" for i in range(X.shape[0])])
        summary = {
            "dataset": "swissnsf",
            "n_before": int(X.shape[0]),
            "n_after": int(X.shape[0]),
            "r_med_before": r_med(X),
            "min_keep_reviews": np.nan,
            "dropped_count": 0,
        }
        X, theta, norm_meta = normalize_to_unit_interval(X, theta)
        summary.update(norm_meta)
        return X, theta, "Swiss NSF (mint_sections means)", ids, summary, []

    if data_name == "beta":
        alpha_theta, beta_theta, kappa = 2.0, 2.0, 20.0
        X, theta = generate_beta_reviews(
            n=args.n, r=args.r,
            alpha_theta=alpha_theta,
            beta_theta=beta_theta,
            kappa=kappa,
            rng=rng,
        )
        ids = np.arange(X.shape[0]).astype(str)
        summary = {
            "dataset": "beta",
            "n_before": int(X.shape[0]),
            "n_after": int(X.shape[0]),
            "r_med_before": r_med(X),
            "min_keep_reviews": np.nan,
            "dropped_count": 0,
            "beta_alpha": alpha_theta,
            "beta_beta": beta_theta,
            "beta_kappa": kappa,
        }
        X, theta, norm_meta = normalize_to_unit_interval(X, theta)
        summary.update(norm_meta)
        return X, theta, f"Beta(2,2), kappa=20 (n={args.n}, r={args.r})", ids, summary, []

    raise ValueError(f"Unknown data source: {data_name}")


# ---------------------------------------------------------------------------
# Main run + plotting
# ---------------------------------------------------------------------------

def save_drop_logs(output_dir: str, summaries: List[dict], dropped_rows: List[dict]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(summaries).to_csv(
        os.path.join(output_dir, "regret_vs_L_cross_drop_summary.csv"), index=False
    )
    pd.DataFrame(dropped_rows).to_csv(
        os.path.join(output_dir, "regret_vs_L_cross_drop_log.csv"), index=False
    )


def run_cross_dataset(args) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)

    base_rng = np.random.default_rng(args.seed)
    datasets = ["beta", "iclr", "neurips", "swissnsf"]

    # Load once for real-data datasets (and one template Beta synthetic set for metadata).
    loaded = {}
    summary_rows = []
    dropped_rows = []
    for name in datasets:
        X, theta, label, ids, summary, dropped = load_dataset(name, args, base_rng)
        loaded[name] = (X, theta, label, ids)
        summary_rows.append(summary)
        dropped_rows.extend(dropped)

    save_drop_logs(args.output_dir, summary_rows, dropped_rows)
    print(f"Saved drop logs in {args.output_dir}")

    for k_name, k_fn in k_configs():
        print(f"\nCross-dataset run: {k_name}")
        panel_results = []
        subtitles = []

        for data_name in datasets:
            X0, theta0, label, _ = loaded[data_name]
            n = X0.shape[0]
            k = k_fn(n)
            rm = r_med(X0)
            rmin = r_min(X0)
            L_values = np.linspace(0.5 * (1.0 / rmin), 1.0, args.n_L_points)
            n_trials = args.beta_trials if data_name == "beta" else 1

            print(f"  {data_name}: n={n}, k={k}, r_min={rmin}, r_med={rm}, trials={n_trials}")
            trials = []
            for _ in range(n_trials):
                if data_name == "beta":
                    trial_rng = np.random.default_rng(base_rng.integers(2**31))
                    X, theta, _, _, _, _ = load_dataset("beta", args, trial_rng)
                    run_rng = np.random.default_rng(base_rng.integers(2**31))
                else:
                    X, theta = X0, theta0
                    run_rng = base_rng

                trials.append(
                    run_regret_vs_L(
                        X,
                        k,
                        L_values,
                        theta=theta,
                        n_softmax_samples=args.n_softmax_samples,
                        rng=run_rng,
                    )
                )

            res = aggregate_trials(trials)
            res["k"] = [k] * len(res["L"])
            res["r_min"] = [rmin] * len(res["L"])
            res["r_med"] = [rm] * len(res["L"])

            out_csv = os.path.join(args.output_dir, f"regret_vs_L_cross_{data_name}_{k_name}.csv")
            pd.DataFrame(res).to_csv(out_csv, index=False)
            print(f"    Saved {out_csv}")

            if args.normalize_by_k:
                # Plot regret per accepted item to improve cross-dataset comparability.
                rplot = dict(res)
                rplot["regret_linear"] = (np.array(res["regret_linear"], dtype=float) / k).tolist()
                rplot["regret_softmax"] = (np.array(res["regret_softmax"], dtype=float) / k).tolist()
                if "regret_linear_std" in res:
                    rplot["regret_linear_std"] = (np.array(res["regret_linear_std"], dtype=float) / k).tolist()
                if "regret_softmax_std" in res:
                    rplot["regret_softmax_std"] = (np.array(res["regret_softmax_std"], dtype=float) / k).tolist()
                panel_results.append(rplot)
            else:
                panel_results.append(res)
            subtitles.append(f"{compact_label(label)} (n={n}, r={rm})")

        fig_path = os.path.join(args.fig_dir, f"regret_vs_L_cross_{k_name}.pdf")
        plot_regret_vs_param_sidebyside(
            panel_results,
            "L",
            subtitles=subtitles,
            suptitle="",
            save_path=fig_path,
            logx=False,
            include_bounds=False,
            x_label=r"$L$",
            y_label="Regret / k" if args.normalize_by_k else "Regret",
            ncols=2,
            sharey=False,
        )
        print(f"  Saved {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-dataset regret experiments")
    parser.add_argument("--experiment", choices=["cross_dataset"], default="cross_dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=200, help="Beta synthetic n")
    parser.add_argument("--r", type=int, default=5, help="Beta synthetic reviews per item")
    parser.add_argument("--n_softmax_samples", type=int, default=10000)
    parser.add_argument("--beta_trials", type=int, default=50)
    parser.add_argument("--n_L_points", type=int, default=15)
    parser.add_argument("--normalize_by_k", action="store_true", default=True)
    parser.add_argument("--no_normalize_by_k", action="store_false", dest="normalize_by_k")
    parser.add_argument("--output_dir", default=os.path.join(os.path.dirname(__file__), "results"))
    parser.add_argument("--fig_dir", default=os.path.join(os.path.dirname(__file__), "figures"))
    args = parser.parse_args()

    run_cross_dataset(args)
