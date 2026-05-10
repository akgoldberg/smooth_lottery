#!/usr/bin/env python3
"""Compare smoothness after matching baseline regret.

For each baseline mechanism/dataset/k row from baseline_local_sensitivity.py,
find the smallest Linear Lottery smoothness L whose regret is no larger than
the baseline regret on the same unperturbed instance. Lower smoothness is
better, so this directly asks how much smoothness Linear Lottery needs to
match each existing method's regret.

Outputs:
- experiments/results/matched_regret_smoothness.csv
- experiments/results/matched_regret_tradeoff_curves.csv
- experiments/figures/matched_regret_smoothness_<k_name>.pdf
- experiments/figures/matched_regret_max_delta_p_<k_name>.pdf
- experiments/figures/matched_regret_tradeoff_<k_name>.pdf
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_BASE = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, _BASE)

import baseline_local_sensitivity as bls
from mechanisms import linear_lottery_mechanism
from utils import k_from_name


BASELINE_MECHANISMS = ["MERIT", "Swiss NSF", "Randomized Threshold"]
DEFAULT_MECHANISMS = ["MERIT", "Swiss NSF"]
DATASET_ORDER = ["Beta", "ICLR", "NeurIPS", "Swiss NSF"]
DEFAULT_DATASETS = ["ICLR", "NeurIPS", "Swiss NSF"]
DATASET_KEYS = {
    "Beta": "beta",
    "ICLR": "iclr",
    "NeurIPS": "neurips",
    "Swiss NSF": "swissnsf",
}
COLORS = {
    "baseline": "#7a7a7a",
    "linear": "#d62728",
    "ratio": "#2ca02c",
    "MERIT": "#2ca02c",
    "Swiss NSF": "#9467bd",
}
HATCHES = {
    "baseline": "////",
    "linear": "xx",
}
DATASET_DISPLAY = {
    "ICLR": "ICLR",
    "NeurIPS": "NeurIPS",
    "Swiss NSF": "Swiss NSF Funding Data",
}
MECHANISM_DISPLAY = {
    "MERIT": "MERIT",
    "Swiss NSF": "Swiss NSF",
}


def parse_csv_list(spec: str) -> List[str]:
    return [x.strip() for x in spec.split(",") if x.strip()]


def display_to_key(spec: str) -> List[str]:
    aliases = {
        "beta": "Beta",
        "iclr": "ICLR",
        "neurips": "NeurIPS",
        "swissnsf": "Swiss NSF",
        "swiss_nsf": "Swiss NSF",
        "swiss nsf": "Swiss NSF",
    }
    out = []
    for raw in parse_csv_list(spec):
        label = aliases.get(raw.lower(), raw)
        if label not in DATASET_ORDER:
            raise ValueError(f"Unknown dataset '{raw}'. Allowed: {DATASET_ORDER}")
        out.append(label)
    return out


def maybe_run_baseline_summary(args) -> str:
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


def load_instances(args) -> Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray, Dict, int]]:
    out = {}
    k_names = parse_csv_list(args.k_names)
    for i, label in enumerate(DATASET_ORDER):
        key = DATASET_KEYS[label]
        rng = np.random.default_rng(args.seed + 1000 * (i + 1))
        X, dlabel, meta = bls.load_dataset(key, args, rng=rng)
        if dlabel != label:
            raise ValueError(f"Dataset label mismatch: expected {label}, got {dlabel}")
        v = bls.utility_vector(X)
        for k_name in k_names:
            k = k_from_name(X.shape[0], k_name)
            out[(label, k_name)] = (X, v, meta, k)
    return out


def linear_regret_for_L(X: np.ndarray, v: np.ndarray, k: int, L: float) -> float:
    p = linear_lottery_mechanism(X, k=k, L=L)
    return bls.expected_regret(v, p, k)


def linear_perturbation_stats_for_saved_edit(
    X: np.ndarray,
    k: int,
    L: float,
    item_idx: int,
    review_col: int,
    direction: int,
    tick: float,
) -> Dict[str, float]:
    """Linear Lottery effect under the baseline-selected one-review edit."""
    p0 = linear_lottery_mechanism(X, k=k, L=L)
    if item_idx < 0 or review_col < 0 or direction == 0:
        return {
            "linear_l1_prob_change_at_required_L": np.nan,
            "linear_max_prob_change_at_required_L": np.nan,
            "linear_local_sensitivity_at_required_L": np.nan,
        }

    Xp = X.copy()
    x0 = float(Xp[item_idx, review_col])
    x1 = float(np.clip(x0 + float(direction) * tick, 0.0, 1.0))
    d_in = abs(x1 - x0)
    if d_in <= 0:
        return {
            "linear_l1_prob_change_at_required_L": np.nan,
            "linear_max_prob_change_at_required_L": np.nan,
            "linear_local_sensitivity_at_required_L": np.nan,
        }

    Xp[item_idx, review_col] = x1
    p1 = linear_lottery_mechanism(Xp, k=k, L=L)
    delta = np.abs(p1 - p0)
    l1 = float(delta.sum())
    return {
        "linear_l1_prob_change_at_required_L": l1,
        "linear_max_prob_change_at_required_L": float(delta.max()),
        "linear_local_sensitivity_at_required_L": l1 / d_in,
        "linear_perturb_item_idx": int(item_idx),
        "linear_perturb_review_col": int(review_col),
        "linear_perturb_direction": int(direction),
    }


def empty_linear_perturbation_stats() -> Dict[str, float]:
    return {
        "linear_l1_prob_change_at_required_L": np.nan,
        "linear_max_prob_change_at_required_L": np.nan,
        "linear_local_sensitivity_at_required_L": np.nan,
        "linear_perturb_item_idx": -1,
        "linear_perturb_review_col": -1,
        "linear_perturb_direction": 0,
    }


def find_linear_L_for_regret(
    X: np.ndarray,
    v: np.ndarray,
    k: int,
    target_regret: float,
    L_min: float,
    L_max: float,
    tol_regret: float,
    tol_L_rel: float,
    max_iter: int,
) -> Tuple[float, float, bool]:
    """Smallest L found with Linear Lottery regret <= target_regret."""
    target = max(0.0, float(target_regret))
    lo = max(float(L_min), 1e-12)
    hi = max(float(L_max), lo)

    reg_hi = linear_regret_for_L(X, v, k, hi)
    while reg_hi > target + tol_regret and hi < 1e12:
        hi *= 2.0
        reg_hi = linear_regret_for_L(X, v, k, hi)

    matched = reg_hi <= target + tol_regret
    if not matched:
        return hi, reg_hi, False

    reg_lo = linear_regret_for_L(X, v, k, lo)
    if reg_lo <= target + tol_regret:
        return lo, reg_lo, True

    best_L = hi
    best_regret = reg_hi
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        reg_mid = linear_regret_for_L(X, v, k, mid)
        if reg_mid <= target + tol_regret:
            hi = mid
            best_L = mid
            best_regret = reg_mid
        else:
            lo = mid
        if abs(hi - lo) <= tol_L_rel * max(1.0, hi):
            break
    return float(best_L), float(best_regret), True


def build_rows(summary: pd.DataFrame, args) -> pd.DataFrame:
    mechanisms = parse_csv_list(args.mechanisms)
    datasets = display_to_key(args.datasets)
    summary = summary[
        summary["mechanism"].isin(mechanisms)
        & summary["dataset"].isin(datasets)
    ].copy()
    instances = load_instances(args)
    rows = []

    for _, row in summary.iterrows():
        dataset = str(row["dataset"])
        k_name = str(row["k_name"])
        mechanism = str(row["mechanism"])
        X, v, meta, k = instances[(dataset, k_name)]

        baseline_regret = float(row["expected_regret"])
        baseline_smoothness = float(row["local_sensitivity"])
        L_req, linear_regret, matched = find_linear_L_for_regret(
            X=X,
            v=v,
            k=k,
            target_regret=baseline_regret,
            L_min=args.L_min,
            L_max=max(args.L_max, baseline_smoothness, float(row.get("L_ref", 0.0))),
            tol_regret=args.tol_regret,
            tol_L_rel=args.tol_L_rel,
            max_iter=args.max_iter,
        )
        ratio = baseline_smoothness / L_req if L_req > 0 else np.inf
        lin_stats = linear_perturbation_stats_for_saved_edit(
            X=X,
            k=k,
            L=L_req,
            item_idx=int(row["item_idx"]),
            review_col=int(row["review_col"]),
            direction=int(row["direction"]),
            tick=float(meta["normalized_tick_size"]),
        )
        rows.append(
            {
                "dataset": dataset,
                "k_name": k_name,
                "k": int(k),
                "n": int(X.shape[0]),
                "baseline_mechanism": mechanism,
                "baseline_regret": baseline_regret,
                "baseline_regret_per_k": baseline_regret / float(k),
                "baseline_smoothness": baseline_smoothness,
                "baseline_l1_prob_change": float(row["l1_prob_change"]),
                "baseline_max_prob_change": float(row["max_prob_change"]),
                "linear_required_L": L_req,
                "linear_regret_at_required_L": linear_regret,
                "linear_regret_per_k_at_required_L": linear_regret / float(k),
                "matched": bool(matched),
                "smoothness_ratio_baseline_over_linear": ratio,
                "log10_smoothness_ratio": float(np.log10(ratio)) if ratio > 0 and np.isfinite(ratio) else np.nan,
                **lin_stats,
            }
        )
    return pd.DataFrame(rows)


def _linear_curve_grid(anchors: List[float], n_points: int) -> np.ndarray:
    vals = np.asarray([float(x) for x in anchors if np.isfinite(x) and float(x) > 0], dtype=float)
    if vals.size == 0:
        vals = np.asarray([1.0])
    lo = max(1e-3, 0.35 * float(vals.min()))
    hi = max(1.0, 1.35 * float(vals.max()))
    if hi / lo > 25:
        base = np.geomspace(lo, hi, n_points)
    else:
        base = np.linspace(lo, hi, n_points)
    grid = np.unique(np.sort(np.concatenate([base, vals])))
    return grid


def build_linear_tradeoff_curves(matched_df: pd.DataFrame, args) -> pd.DataFrame:
    """Compute Linear Lottery regret/k curves over an adaptive L grid."""
    instances = load_instances(args)
    rows = []
    for dataset in display_to_key(args.datasets):
        for k_name in parse_csv_list(args.k_names):
            block = matched_df[(matched_df["dataset"] == dataset) & (matched_df["k_name"] == k_name)]
            if block.empty:
                continue
            X, v, _, k = instances[(dataset, k_name)]
            anchors = (
                block["baseline_smoothness"].astype(float).tolist()
                + block["linear_required_L"].astype(float).tolist()
                + [float(block["baseline_smoothness"].median())]
            )
            for L in _linear_curve_grid(anchors, n_points=args.tradeoff_L_points):
                regret = linear_regret_for_L(X, v, k, float(L))
                rows.append(
                    {
                        "dataset": dataset,
                        "k_name": k_name,
                        "k": int(k),
                        "n": int(X.shape[0]),
                        "L": float(L),
                        "linear_regret": float(regret),
                        "linear_regret_per_k": float(regret / float(k)),
                    }
                )
    return pd.DataFrame(rows)


def plot_rows(df: pd.DataFrame, fig_dir: str) -> List[str]:
    os.makedirs(fig_dir, exist_ok=True)
    plt.rcParams.update({
        "font.size": 15,
        "axes.labelsize": 17,
        "axes.titlesize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    })
    out = []
    for k_name in sorted(df["k_name"].unique()):
        dk = df[df["k_name"] == k_name].copy()
        dataset_order = [d for d in DEFAULT_DATASETS if d in set(dk["dataset"])]
        mechanism_order = [m for m in DEFAULT_MECHANISMS if m in set(dk["baseline_mechanism"])]
        groups = []
        for dataset in dataset_order:
            for mech in mechanism_order:
                block = dk[(dk["dataset"] == dataset) & (dk["baseline_mechanism"] == mech)]
                if len(block):
                    groups.append((dataset, mech, block.iloc[0]))

        fig, ax = plt.subplots(figsize=(13.5, 5.8))
        x = np.arange(len(groups), dtype=float)
        width = 0.34
        baseline_vals = [float(r["baseline_smoothness"]) for _, _, r in groups]
        linear_vals = [float(r["linear_required_L"]) for _, _, r in groups]
        ratios = [float(r["smoothness_ratio_baseline_over_linear"]) for _, _, r in groups]
        regret_per_k = [float(r["baseline_regret_per_k"]) for _, _, r in groups]
        all_vals = np.asarray(baseline_vals + linear_vals, dtype=float)
        sorted_vals = np.sort(all_vals[np.isfinite(all_vals)])
        if len(sorted_vals) >= 2 and sorted_vals[-1] > 5.0 * sorted_vals[-2]:
            y_cap = 1.25 * sorted_vals[-2]
        else:
            y_cap = 1.22 * sorted_vals[-1] if len(sorted_vals) else 1.0
        y_cap = max(y_cap, 1.0)

        baseline_plot_vals = [min(v, 0.96 * y_cap) for v in baseline_vals]
        linear_plot_vals = [min(v, 0.96 * y_cap) for v in linear_vals]
        has_truncated_bar = any(
            (v > vp)
            for v, vp in zip(baseline_vals + linear_vals, baseline_plot_vals + linear_plot_vals)
        )

        ax.bar(
            x - width / 2,
            baseline_plot_vals,
            width,
            color=COLORS["baseline"],
            edgecolor="black",
            linewidth=0.8,
            hatch=HATCHES["baseline"],
            label="Existing method empirical smoothness",
        )
        ax.bar(
            x + width / 2,
            linear_plot_vals,
            width,
            color=COLORS["linear"],
            edgecolor="black",
            linewidth=0.8,
            hatch=HATCHES["linear"],
            label="Linear Lottery smoothness at matched regret",
        )

        ax.set_ylim(0, y_cap)
        for xpos, y0, y1, y0_plot, y1_plot, ratio in zip(
            x, baseline_vals, linear_vals, baseline_plot_vals, linear_plot_vals, ratios
        ):
            if np.isfinite(ratio):
                ax.text(
                    xpos - width / 2,
                    min(max(y0_plot, y1_plot) + y_cap * 0.035, y_cap * 0.97),
                    f"{ratio:.1f}x",
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5),
                )
            if y0 > y0_plot:
                ax.text(
                    xpos - width / 2,
                    y0_plot,
                    f"...\n{y0:.0f}\ntruncated",
                    ha="center",
                    va="bottom",
                    fontsize=11,
                    fontweight="bold",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.2),
                )
            if y1 > y1_plot:
                ax.text(
                    xpos + width / 2,
                    y1_plot,
                    f"...\n{y1:.0f}\ntruncated",
                    ha="center",
                    va="bottom",
                    fontsize=11,
                    fontweight="bold",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.2),
                )

        def fmt_regret(x: float) -> str:
            if x == 0:
                return "0"
            if abs(x) < 0.001:
                return f"{x:.1e}"
            return f"{x:.3f}"

        labels = [
            f"{MECHANISM_DISPLAY.get(mech, mech)}\n$\\it{{regret/k={fmt_regret(rk)}}}$"
            for (_, mech, _), rk in zip(groups, regret_per_k)
        ]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0)
        ylabel = "Smoothness (lower is better)"
        if has_truncated_bar:
            ylabel += "\n(truncated bars show true value)"
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)

        # Vertical separators and dataset labels.
        cursor = 0
        for dataset in dataset_order:
            count = sum(1 for d, _, _ in groups if d == dataset)
            if count == 0:
                continue
            center = cursor + (count - 1) / 2
            ax.text(
                center,
                -0.16,
                DATASET_DISPLAY.get(dataset, dataset),
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=15,
                fontweight="bold",
            )
            cursor += count
            if cursor < len(groups):
                ax.axvline(cursor - 0.5, color="#333333", linewidth=0.9)

        ax.legend(loc="best", frameon=True)
        fig.tight_layout()
        out_pdf = os.path.join(fig_dir, f"matched_regret_smoothness_{k_name}.pdf")
        fig.savefig(out_pdf, bbox_inches="tight")
        plt.close(fig)
        out.append(out_pdf)
    return out


def plot_max_delta_p_rows(df: pd.DataFrame, fig_dir: str) -> List[str]:
    os.makedirs(fig_dir, exist_ok=True)
    out = []
    for k_name in sorted(df["k_name"].unique()):
        dk = df[df["k_name"] == k_name].copy()
        dataset_order = [d for d in DEFAULT_DATASETS if d in set(dk["dataset"])]
        mechanism_order = [m for m in DEFAULT_MECHANISMS if m in set(dk["baseline_mechanism"])]
        groups = []
        for dataset in dataset_order:
            for mech in mechanism_order:
                block = dk[(dk["dataset"] == dataset) & (dk["baseline_mechanism"] == mech)]
                if len(block):
                    groups.append((dataset, mech, block.iloc[0]))

        fig, ax = plt.subplots(figsize=(13.5, 5.8))
        x = np.arange(len(groups), dtype=float)
        width = 0.34
        baseline_vals = [float(r["baseline_max_prob_change"]) for _, _, r in groups]
        linear_vals = [float(r["linear_max_prob_change_at_required_L"]) for _, _, r in groups]
        regret_per_k = [float(r["baseline_regret_per_k"]) for _, _, r in groups]

        ax.bar(
            x - width / 2,
            baseline_vals,
            width,
            color=COLORS["baseline"],
            edgecolor="black",
            linewidth=0.8,
            hatch=HATCHES["baseline"],
            label=r"Existing method max $\Delta p_i$",
        )
        ax.bar(
            x + width / 2,
            linear_vals,
            width,
            color=COLORS["linear"],
            edgecolor="black",
            linewidth=0.8,
            hatch=HATCHES["linear"],
            label=r"Linear Lottery max $\Delta p_i$ at matched regret",
        )

        ymax = max(max(baseline_vals, default=0.0), max(linear_vals, default=0.0))
        ax.set_ylim(0, ymax * 1.15 if ymax > 0 else 1.0)
        def fmt_regret(x: float) -> str:
            if x == 0:
                return "0"
            if abs(x) < 0.001:
                return f"{x:.1e}"
            return f"{x:.3f}"

        labels = [
            f"{MECHANISM_DISPLAY.get(mech, mech)}\n$\\it{{regret/k={fmt_regret(rk)}}}$"
            for (_, mech, _), rk in zip(groups, regret_per_k)
        ]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0)
        ax.set_ylabel("Max change in probability")
        ax.grid(axis="y", alpha=0.25)

        cursor = 0
        for dataset in dataset_order:
            count = sum(1 for d, _, _ in groups if d == dataset)
            if count == 0:
                continue
            center = cursor + (count - 1) / 2
            ax.text(
                center,
                -0.12,
                DATASET_DISPLAY.get(dataset, dataset),
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=15,
                fontweight="bold",
            )
            cursor += count
            if cursor < len(groups):
                ax.axvline(cursor - 0.5, color="#333333", linewidth=0.9)

        ax.legend(loc="best", frameon=True)
        fig.tight_layout()
        out_pdf = os.path.join(fig_dir, f"matched_regret_max_delta_p_{k_name}.pdf")
        fig.savefig(out_pdf, bbox_inches="tight")
        plt.close(fig)
        out.append(out_pdf)
    return out


def plot_tradeoff_rows(matched_df: pd.DataFrame, curve_df: pd.DataFrame, fig_dir: str) -> List[str]:
    os.makedirs(fig_dir, exist_ok=True)
    out = []
    marker_map = {"MERIT": "o", "Swiss NSF": "s"}
    for k_name in sorted(matched_df["k_name"].unique()):
        dk = matched_df[matched_df["k_name"] == k_name].copy()
        ck = curve_df[curve_df["k_name"] == k_name].copy()
        dataset_order = [d for d in DEFAULT_DATASETS if d in set(dk["dataset"])]
        if not dataset_order:
            continue

        fig, axes = plt.subplots(
            1,
            len(dataset_order),
            figsize=(7.6 * len(dataset_order), 7.6),
            sharey=False,
            constrained_layout=False,
        )
        if len(dataset_order) == 1:
            axes = [axes]

        legend_handles = None
        legend_labels = None
        for ax, dataset in zip(axes, dataset_order):
            csub = ck[ck["dataset"] == dataset].sort_values("L")
            ax.plot(
                csub["L"].to_numpy(),
                csub["linear_regret_per_k"].to_numpy(),
                color=COLORS["linear"],
                linewidth=4.4,
                label="Linear Lottery",
            )

            bsub = dk[dk["dataset"] == dataset]
            for mech in DEFAULT_MECHANISMS:
                row = bsub[bsub["baseline_mechanism"] == mech]
                if row.empty:
                    continue
                r = row.iloc[0]
                ax.scatter(
                    [float(r["baseline_smoothness"])],
                    [float(r["baseline_regret_per_k"])],
                    color=COLORS.get(mech, "#333333"),
                    marker=marker_map.get(mech, "o"),
                    s=220,
                    edgecolor="black",
                    linewidth=1.3,
                    zorder=4,
                    label=MECHANISM_DISPLAY.get(mech, mech),
                )
                ax.scatter(
                    [float(r["linear_required_L"])],
                    [float(r["linear_regret_per_k_at_required_L"])],
                    facecolor="white",
                    edgecolor=COLORS["linear"],
                    marker=marker_map.get(mech, "o"),
                    s=190,
                    linewidth=2.8,
                    zorder=5,
                    label="_nolegend_",
                )

            x_vals = np.concatenate([
                csub["L"].to_numpy(dtype=float),
                bsub["baseline_smoothness"].to_numpy(dtype=float),
            ])
            x_vals = x_vals[np.isfinite(x_vals) & (x_vals > 0)]
            if len(x_vals) and x_vals.max() / x_vals.min() > 25:
                ax.set_xscale("log")
            ax.set_title(DATASET_DISPLAY.get(dataset, dataset), fontsize=34, pad=18)
            ax.set_xlabel("$L$ (lower is smoother)", fontsize=30, labelpad=10)
            if ax is axes[0]:
                ax.set_ylabel("Regret / k", fontsize=30, labelpad=12)
            ax.tick_params(axis="both", labelsize=23)
            ax.grid(alpha=0.25, linestyle=":")
            ax.set_ylim(bottom=0)
            if legend_handles is None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()

        if legend_handles:
            dedup = dict(zip(legend_labels, legend_handles))
            fig.legend(
                dedup.values(),
                dedup.keys(),
                loc="upper center",
                bbox_to_anchor=(0.5, 0.975),
                ncol=len(dedup),
                frameon=True,
                fontsize=28,
            )
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.85])
        out_pdf = os.path.join(fig_dir, f"matched_regret_tradeoff_{k_name}.pdf")
        fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        out.append(out_pdf)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoothness comparison at matched baseline regret")
    parser.add_argument("--k_names", default="k10pct,k33pct,k50pct")
    parser.add_argument("--datasets", default="ICLR,NeurIPS,Swiss NSF")
    parser.add_argument("--mechanisms", default="MERIT,Swiss NSF")
    parser.add_argument("--interval_method", default="leave_one_out", choices=["leave_one_out", "gaussian_ci", "minmax"])
    parser.add_argument("--candidate_window", type=int, default=0)
    parser.add_argument("--swiss_candidate_window", type=int, default=-1)
    parser.add_argument("--threshold_candidate_window", type=int, default=-1)
    parser.add_argument("--threshold_band_frac", type=float, default=0.10)
    parser.add_argument("--rerun_baseline", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--L_min", type=float, default=1e-6)
    parser.add_argument("--L_max", type=float, default=1.0)
    parser.add_argument("--tol_regret", type=float, default=1e-10)
    parser.add_argument("--tol_L_rel", type=float, default=1e-4)
    parser.add_argument("--max_iter", type=int, default=80)
    parser.add_argument("--tradeoff_L_points", type=int, default=36)

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
    df = build_rows(summary, args)
    out_csv = os.path.join(args.output_dir, "matched_regret_smoothness.csv")
    df.to_csv(out_csv, index=False)
    curve_df = build_linear_tradeoff_curves(df, args)
    out_curve_csv = os.path.join(args.output_dir, "matched_regret_tradeoff_curves.csv")
    curve_df.to_csv(out_curve_csv, index=False)
    fig_paths = plot_rows(df, args.fig_dir)
    max_dp_fig_paths = plot_max_delta_p_rows(df, args.fig_dir)
    tradeoff_fig_paths = plot_tradeoff_rows(df, curve_df, args.fig_dir)

    print(f"Saved {out_csv}")
    print(f"Saved {out_curve_csv}")
    for path in fig_paths + max_dp_fig_paths + tradeoff_fig_paths:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
