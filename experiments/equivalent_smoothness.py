#!/usr/bin/env python3
"""Compare existing baselines to Linear Lottery at matched smoothness.

The existing baseline mechanisms do not expose the paper's smoothness
parameter L. This script uses the empirical local sensitivity measured by
baseline_local_sensitivity.py as their smoothness level, then evaluates Linear
Lottery at that same L on the same dataset/k instance.

Outputs:
- experiments/results/equivalent_smoothness.csv
- experiments/figures/equivalent_smoothness_<k_name>.pdf
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

_BASE = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, _BASE)

import baseline_local_sensitivity as bls
from mechanisms import linear_lottery_mechanism
from utils import k_from_name


BASELINE_MECHANISMS = ["MERIT", "Swiss NSF", "Randomized Threshold"]
DATASET_ORDER = ["Beta", "ICLR", "NeurIPS", "Swiss NSF"]
DATASET_KEYS = {
    "Beta": "beta",
    "ICLR": "iclr",
    "NeurIPS": "neurips",
    "Swiss NSF": "swissnsf",
}
BASELINE_COLORS = {
    "MERIT": "#2ca02c",
    "Swiss NSF": "#d62728",
    "Randomized Threshold": "#9467bd",
}
BASELINE_MARKERS = {
    "MERIT": "o",
    "Swiss NSF": "s",
    "Randomized Threshold": "^",
}
LINEAR_COLOR = "#1f77b4"


def parse_csv_list(spec: str) -> List[str]:
    return [x.strip() for x in spec.split(",") if x.strip()]


def maybe_run_baseline_summary(args) -> str:
    """Run baseline_local_sensitivity.py if requested or if summary is absent."""
    summary_path = os.path.join(args.output_dir, "baseline_local_sensitivity_summary.csv")
    if os.path.exists(summary_path) and not args.rerun_baseline:
        return summary_path

    run_args = argparse.Namespace(
        k_names=args.k_names,
        interval_method=args.interval_method,
        candidate_window=args.candidate_window,
        swiss_candidate_window=args.swiss_candidate_window,
        threshold_candidate_window=args.threshold_candidate_window,
        threshold_band_frac=args.threshold_band_frac,
        mechanisms=",".join(BASELINE_MECHANISMS),
        softmax_samples=0,
        softmax_reps=0,
        seed=args.seed,
        beta_n=args.beta_n,
        beta_r=args.beta_r,
        beta_alpha=args.beta_alpha,
        beta_beta=args.beta_beta,
        beta_kappa=args.beta_kappa,
        output_dir=args.output_dir,
        fig_dir=args.fig_dir,
    )
    df, summary = bls.run(run_args)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)
    df.to_csv(os.path.join(args.output_dir, "baseline_local_sensitivity.csv"), index=False)
    summary.to_csv(summary_path, index=False)
    if not summary.empty:
        bls.write_baseline_tables_tex(summary, fig_dir=args.fig_dir)
    return summary_path


def load_linear_instances(args) -> Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray, int]]:
    """Load each dataset once per requested k, matching baseline setup."""
    out = {}
    k_names = parse_csv_list(args.k_names)
    for i, label in enumerate(DATASET_ORDER):
        key = DATASET_KEYS[label]
        rng = np.random.default_rng(args.seed + 1000 * (i + 1))
        X, dlabel, _ = bls.load_dataset(key, args, rng=rng)
        if dlabel != label:
            raise ValueError(f"Dataset label mismatch: expected {label}, got {dlabel}")
        v = bls.utility_vector(X)
        for k_name in k_names:
            k = k_from_name(X.shape[0], k_name)
            out[(label, k_name)] = (X, v, k)
    return out


def equivalent_rows(summary: pd.DataFrame, args) -> pd.DataFrame:
    summary = summary[summary["mechanism"].isin(BASELINE_MECHANISMS)].copy()
    if summary.empty:
        return pd.DataFrame()

    instances = load_linear_instances(args)
    rows = []
    for _, row in summary.iterrows():
        dataset = str(row["dataset"])
        k_name = str(row["k_name"])
        mechanism = str(row["mechanism"])
        smoothness = float(row["local_sensitivity"])
        if not np.isfinite(smoothness) or smoothness <= 0:
            continue

        X, v, k = instances[(dataset, k_name)]
        p_linear = linear_lottery_mechanism(X, k=k, L=smoothness)
        linear_regret = bls.expected_regret(v, p_linear, k)
        baseline_regret = float(row["expected_regret"])

        rows.append(
            {
                "dataset": dataset,
                "k_name": k_name,
                "k": int(k),
                "n": int(X.shape[0]),
                "baseline_mechanism": mechanism,
                "matched_smoothness": smoothness,
                "baseline_l1_prob_change": float(row["l1_prob_change"]),
                "baseline_max_prob_change": float(row["max_prob_change"]),
                "baseline_regret": baseline_regret,
                "baseline_regret_per_k": baseline_regret / float(k),
                "linear_L": smoothness,
                "linear_regret": float(linear_regret),
                "linear_regret_per_k": float(linear_regret / float(k)),
                "linear_minus_baseline_regret_per_k": float(linear_regret / float(k) - baseline_regret / float(k)),
            }
        )
    return pd.DataFrame(rows)


def plot_equivalent_smoothness(df: pd.DataFrame, fig_dir: str) -> List[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        print(f"[warn] Skipping plots because plotting dependency is unavailable: {exc}")
        return []

    os.makedirs(fig_dir, exist_ok=True)
    out_paths = []
    if df.empty:
        return out_paths

    for k_name in sorted(df["k_name"].unique()):
        dk = df[df["k_name"] == k_name].copy()
        fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), sharex=False, sharey=False)
        axes = axes.ravel()
        for ax, dataset in zip(axes, DATASET_ORDER):
            block = dk[dk["dataset"] == dataset]
            if block.empty:
                ax.set_visible(False)
                continue
            for _, r in block.iterrows():
                mech = str(r["baseline_mechanism"])
                x = float(r["matched_smoothness"])
                y_base = float(r["baseline_regret_per_k"])
                y_lin = float(r["linear_regret_per_k"])
                ax.scatter(
                    [x],
                    [y_base],
                    color=BASELINE_COLORS[mech],
                    marker=BASELINE_MARKERS[mech],
                    s=55,
                    label=mech,
                    zorder=3,
                )
                ax.scatter(
                    [x],
                    [y_lin],
                    color=LINEAR_COLOR,
                    marker="x",
                    s=65,
                    label="Linear Lottery at matched L",
                    zorder=4,
                )
                ax.plot([x, x], [y_base, y_lin], color="#999999", linewidth=1.0, alpha=0.6)
            ax.set_title(dataset)
            ax.set_xlabel("Matched empirical smoothness")
            ax.set_ylabel("Regret / k")
            ax.set_ylim(bottom=0)
            ax.grid(alpha=0.25)
            xs = block["matched_smoothness"].to_numpy(dtype=float)
            if np.nanmax(xs) / max(np.nanmin(xs), 1e-12) > 20:
                ax.set_xscale("log")

        handles, labels = axes[0].get_legend_handles_labels()
        dedup = dict(zip(labels, handles))
        fig.legend(
            dedup.values(),
            dedup.keys(),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=4,
            fontsize=10,
            frameon=True,
        )
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
        out_pdf = os.path.join(fig_dir, f"equivalent_smoothness_{k_name}.pdf")
        fig.savefig(out_pdf, bbox_inches="tight")
        plt.close(fig)
        out_paths.append(out_pdf)
    return out_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline regret vs Linear Lottery at equivalent smoothness")
    parser.add_argument("--k_names", default="k10pct,k33pct,k50pct")
    parser.add_argument("--interval_method", default="leave_one_out", choices=["leave_one_out", "gaussian_ci", "minmax"])
    parser.add_argument("--candidate_window", type=int, default=0)
    parser.add_argument("--swiss_candidate_window", type=int, default=-1)
    parser.add_argument("--threshold_candidate_window", type=int, default=-1)
    parser.add_argument("--threshold_band_frac", type=float, default=0.10)
    parser.add_argument("--rerun_baseline", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--beta_n", type=int, default=200)
    parser.add_argument("--beta_r", type=int, default=5)
    parser.add_argument("--beta_alpha", type=float, default=2.0)
    parser.add_argument("--beta_beta", type=float, default=2.0)
    parser.add_argument("--beta_kappa", type=float, default=100.0)

    parser.add_argument("--output_dir", default=os.path.join(os.path.dirname(__file__), "results"))
    parser.add_argument("--fig_dir", default=os.path.join(os.path.dirname(__file__), "figures"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)

    summary_path = maybe_run_baseline_summary(args)
    summary = pd.read_csv(summary_path)
    df = equivalent_rows(summary, args)
    out_csv = os.path.join(args.output_dir, "equivalent_smoothness.csv")
    df.to_csv(out_csv, index=False)
    fig_paths = plot_equivalent_smoothness(df, args.fig_dir)

    print(f"Saved {out_csv}")
    for path in fig_paths:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
