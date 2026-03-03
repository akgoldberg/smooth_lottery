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
import glob
from types import SimpleNamespace
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd

_BASE = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, _BASE)

from mechanisms import linear_lottery_mechanism, softmax_mechanism
from data_utils import (
    load_review_matrix,
    load_swiss_nsf_point_estimates,
    generate_beta_reviews,
)
from utils import reviews_per_item, r_min, r_med, normalize_scores
from plot_results import plot_regret_vs_param_sidebyside, plot_beta_sweep


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def compute_regret(v: np.ndarray, p: np.ndarray, k: int) -> float:
    top_k_vals = np.sort(v)[-k:]
    return float(top_k_vals.sum() - np.dot(v, p))


def compact_label(label: str) -> str:
    if label.startswith("Beta("):
        return "Beta"
    return (label
            .replace("Swiss NSF (mint_sections means)", "Swiss NSF")
            .replace("NeurIPS 2024", "NeurIPS")
            .replace("ICLR 2025", "ICLR"))


def normalize_to_unit_interval(
    X: np.ndarray, theta: Optional[np.ndarray], dataset_key: str
) -> Tuple[np.ndarray, Optional[np.ndarray], dict]:
    """Scale-based normalization to [0,1] using configured review scales."""
    return normalize_scores(X, dataset_key=dataset_key, theta=theta, synthetic_ticks=10)


def k_configs() -> List[Tuple[str, Callable[[int], int]]]:
    return [
        ("k1", lambda n: 1),
        ("k10pct", lambda n: max(1, min(int(0.1 * n), n - 1))),
        ("k33pct", lambda n: max(1, min(int(round(n / 3.0)), n - 1))),
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
        X, theta, norm_meta = normalize_to_unit_interval(X, theta, dataset_key="neurips2024")
        summary.update(norm_meta)
        return X, theta, "NeurIPS 2024", ids, summary, rows

    if data_name == "iclr":
        X, paper_ids, _ = load_review_matrix("iclr2025", drop_rejected=True)
        X, theta, ids, summary, rows = drop_low_review_outliers(
            X, None, paper_ids, dataset="iclr"
        )
        X, theta, norm_meta = normalize_to_unit_interval(X, theta, dataset_key="iclr2025")
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
        X, theta, norm_meta = normalize_to_unit_interval(X, theta, dataset_key="swissnsf")
        summary.update(norm_meta)
        return X, theta, "Swiss NSF (mint_sections means)", ids, summary, []

    if data_name == "beta":
        alpha_theta, beta_theta, kappa = 2.0, 2.0, float(args.beta_kappa)
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
        X, theta, norm_meta = normalize_to_unit_interval(X, theta, dataset_key="synthetic")
        summary.update(norm_meta)
        return X, theta, f"Beta(2,2), kappa={kappa:g} (n={args.n}, r={args.r})", ids, summary, []

    raise ValueError(f"Unknown data source: {data_name}")


# ---------------------------------------------------------------------------
# Main run + plotting
# ---------------------------------------------------------------------------

def save_drop_logs(output_dir: str, summaries: List[dict], dropped_rows: List[dict]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(summaries).to_csv(
        os.path.join(output_dir, "regret_vs_L_drop_summary.csv"), index=False
    )
    pd.DataFrame(dropped_rows).to_csv(
        os.path.join(output_dir, "regret_vs_L_drop_log.csv"), index=False
    )


def clear_old_cross_outputs(output_dir: str, fig_dir: str) -> None:
    for pattern in [
        os.path.join(output_dir, "regret_vs_L_*.csv"),
        os.path.join(fig_dir, "regret_vs_L_*.pdf"),
    ]:
        for path in glob.glob(pattern):
            os.remove(path)
            print(f"Removed old artifact: {path}")


def clear_old_cross_figures(fig_dir: str) -> None:
    for path in glob.glob(os.path.join(fig_dir, "regret_vs_L_*.pdf")):
        os.remove(path)
        print(f"Removed old artifact: {path}")


def replot_cross_dataset(args) -> None:
    datasets = ["beta", "iclr", "neurips", "swissnsf"]
    k_names = ["k1", "k10pct", "k33pct", "k50pct"]

    os.makedirs(args.fig_dir, exist_ok=True)
    if args.clear_old:
        clear_old_cross_figures(args.fig_dir)

    meta_args = SimpleNamespace(n=args.n, r=args.r, beta_kappa=args.beta_kappa)
    rng = np.random.default_rng(args.seed)
    meta = {}
    for ds in datasets:
        X, _, label, _, _, _ = load_dataset(ds, meta_args, rng)
        meta[ds] = (compact_label(label), X.shape[0], r_med(X))

    for k_name in k_names:
        panel_results = []
        subtitles = []
        for ds in datasets:
            path = os.path.join(args.output_dir, f"regret_vs_L_{ds}_{k_name}.csv")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing input CSV: {path}")
            df = pd.read_csv(path)
            k = float(df["k"].iloc[0])
            res = {c: df[c].tolist() for c in df.columns}
            if args.normalize_by_k:
                res["regret_linear"] = (df["regret_linear"] / k).tolist()
                res["regret_softmax"] = (df["regret_softmax"] / k).tolist()
                if "regret_linear_std" in df.columns:
                    res["regret_linear_std"] = (df["regret_linear_std"] / k).tolist()
                if "regret_softmax_std" in df.columns:
                    res["regret_softmax_std"] = (df["regret_softmax_std"] / k).tolist()

            label, n, r = meta[ds]
            subtitles.append(f"{label} (n={n}, r={r})")
            panel_results.append(res)

        fig_path = os.path.join(args.fig_dir, f"regret_vs_L_{k_name}.pdf")
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
        print(f"Saved {fig_path}")


def run_cross_dataset(args) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)
    if args.clear_old:
        clear_old_cross_outputs(args.output_dir, args.fig_dir)

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

            out_csv = os.path.join(args.output_dir, f"regret_vs_L_{data_name}_{k_name}.csv")
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

        fig_path = os.path.join(args.fig_dir, f"regret_vs_L_{k_name}.pdf")
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


# ---------------------------------------------------------------------------
# Symmetric Beta sweep (merged from regret_beta_sweep.py)
# ---------------------------------------------------------------------------

def parse_l_multipliers(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _beta_sweep_k_fn(k_name: str):
    if k_name == "k10pct":
        return lambda n: max(1, min(int(0.1 * n), n - 1))
    if k_name == "k33pct":
        return lambda n: max(1, min(int(round(n / 3.0)), n - 1))
    if k_name == "k50pct":
        return lambda n: max(1, min(int(0.5 * n), n - 1))
    raise ValueError(f"Unknown k_name for beta sweep: {k_name}")


def run_beta_symmetric_sweep(args, k_name: str) -> pd.DataFrame:
    rows = []
    k_fn = _beta_sweep_k_fn(k_name)
    for alpha in range(args.beta_sweep_alpha_min, args.beta_sweep_alpha_max + 1):
        for trial in range(args.beta_sweep_trials):
            rng = np.random.default_rng(args.seed + 10000 * alpha + trial)
            X, theta = generate_beta_reviews(
                n=args.n,
                r=args.r,
                alpha_theta=float(alpha),
                beta_theta=float(alpha),
                kappa=args.beta_kappa,
                rng=rng,
            )
            Xn, theta_n, _ = normalize_scores(X, dataset_key="synthetic", theta=theta, synthetic_ticks=10)
            v = theta_n if theta_n is not None else np.nanmean(Xn, axis=1)
            n_items = Xn.shape[0]
            k = k_fn(n_items)
            r_value = args.r

            for L_mult in args.beta_sweep_L_multipliers:
                L = float(L_mult * (1.0 / r_value))
                p_lin = linear_lottery_mechanism(Xn, k, L)
                p_soft = softmax_mechanism(
                    Xn,
                    k,
                    L,
                    n_samples=args.beta_sweep_softmax_samples,
                    rng=rng,
                )
                rows.append(
                    {
                        "k_name": k_name,
                        "alpha": alpha,
                        "beta": alpha,
                        "trial": trial,
                        "n": n_items,
                        "r": r_value,
                        "k": k,
                        "L_multiplier": L_mult,
                        "L": L,
                        "regret_linear": compute_regret(v, p_lin, k),
                        "regret_softmax": compute_regret(v, p_soft, k),
                    }
                )
    df = pd.DataFrame(rows)
    df["regret_linear_per_k"] = df["regret_linear"] / df["k"]
    df["regret_softmax_per_k"] = df["regret_softmax"] / df["k"]
    return df


def run_beta_sweep_experiment(args) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)
    k_names = [x.strip() for x in args.beta_sweep_k_names.split(",") if x.strip()]
    for k_name in k_names:
        df = run_beta_symmetric_sweep(args, k_name=k_name)
        out_csv = os.path.join(args.output_dir, f"regret_beta_sweep_{k_name}.csv")
        out_pdf = os.path.join(args.fig_dir, f"regret_beta_sweep_{k_name}.pdf")
        df.to_csv(out_csv, index=False)
        plot_beta_sweep(df, out_pdf=out_pdf, k_name=k_name)
        print(f"Saved {out_csv}")
        print(f"Saved {out_pdf}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-dataset regret experiments")
    parser.add_argument("--experiment", choices=["cross_dataset"], default="cross_dataset")
    parser.add_argument("--plot_only", action="store_true",
                        help="Regenerate regret figures from existing CSVs without rerunning mechanisms.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=200, help="Beta synthetic n")
    parser.add_argument("--r", type=int, default=5, help="Beta synthetic reviews per item")
    parser.add_argument("--beta_kappa", type=float, default=100.0,
                        help="Beta reviewer-noise concentration (higher => lower reviewer noise).")
    parser.add_argument("--n_softmax_samples", type=int, default=10000)
    parser.add_argument("--beta_trials", type=int, default=50)
    parser.add_argument("--n_L_points", type=int, default=15)
    parser.add_argument("--normalize_by_k", action="store_true", default=True)
    parser.add_argument("--no_normalize_by_k", action="store_false", dest="normalize_by_k")
    parser.add_argument("--clear_old", action="store_true", default=True)
    parser.add_argument("--no_clear_old", action="store_false", dest="clear_old")
    parser.add_argument("--output_dir", default=os.path.join(os.path.dirname(__file__), "results"))
    parser.add_argument("--fig_dir", default=os.path.join(os.path.dirname(__file__), "figures"))
    parser.add_argument("--run_beta_sweep", action="store_true", default=True)
    parser.add_argument("--no_run_beta_sweep", action="store_false", dest="run_beta_sweep")
    parser.add_argument("--beta_sweep_alpha_min", type=int, default=1)
    parser.add_argument("--beta_sweep_alpha_max", type=int, default=10)
    parser.add_argument("--beta_sweep_k_names", default="k10pct,k33pct,k50pct")
    parser.add_argument("--beta_sweep_L_multipliers", default="1.0,2.0")
    parser.add_argument("--beta_sweep_trials", type=int, default=20)
    parser.add_argument("--beta_sweep_softmax_samples", type=int, default=6000)
    args = parser.parse_args()
    args.beta_sweep_L_multipliers = parse_l_multipliers(args.beta_sweep_L_multipliers)

    if args.plot_only:
        replot_cross_dataset(args)
    else:
        run_cross_dataset(args)
        if args.run_beta_sweep:
            run_beta_sweep_experiment(args)
