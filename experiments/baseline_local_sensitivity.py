#!/usr/bin/env python3
"""Worst-case local sensitivity for existing baseline partial-lottery mechanisms.

Baselines:
- MERIT (interval LP)
- Swiss NSF interval rule
- Randomize-above-threshold

No Monte Carlo is used. For interval-based mechanisms, perturbations shift the
entire item interval and point estimate together (lower, midpoint/mean, upper).
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

_BASE = os.path.dirname(os.path.dirname(__file__))
_BASELINES_ROOT = (
    os.path.join(_BASE, "merit_baselines")
    if os.path.isdir(os.path.join(_BASE, "merit_baselines"))
    else os.path.join(_BASE, "baselines")
)
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, _BASE)
sys.path.insert(0, _BASELINES_ROOT)

from algorithm.helpers import swiss_nsf as swiss_nsf_intervals
from data_utils import generate_beta_reviews, load_review_matrix, load_swiss_nsf_point_estimates
from mechanisms import randomize_above_threshold, reviews_to_intervals
from utils import drop_low_review_outliers, k_from_name, normalize_scores, r_min


def _run_merit(intervals: List[Tuple[float, float]], k: int) -> np.ndarray:
    # Lazy import so script can still run Swiss/threshold analysis if Gurobi/MERIT is unavailable.
    from algorithm.merit import run_merit

    p, _ = run_merit(intervals, k)
    return np.asarray(p, dtype=float)


def load_dataset(name: str, args, rng: np.random.Generator) -> Tuple[np.ndarray, str, Dict]:
    if name == "iclr":
        X, ids, _ = load_review_matrix("iclr2025", drop_rejected=True)
        X, _ = drop_low_review_outliers(X, ids)
        Xn, _, meta = normalize_scores(X, dataset_key="iclr2025", synthetic_ticks=10)
        return Xn, "ICLR", meta
    if name == "neurips":
        X, ids, _ = load_review_matrix("neurips2024", drop_rejected=True)
        X, _ = drop_low_review_outliers(X, ids)
        Xn, _, meta = normalize_scores(X, dataset_key="neurips2024", synthetic_ticks=10)
        return Xn, "NeurIPS", meta
    if name == "swissnsf":
        X, theta = load_swiss_nsf_point_estimates()
        Xn, _, meta = normalize_scores(X, dataset_key="swissnsf", theta=theta, synthetic_ticks=10)
        return Xn, "Swiss NSF", meta
    if name == "beta":
        X, theta = generate_beta_reviews(
            n=args.beta_n,
            r=args.beta_r,
            alpha_theta=args.beta_alpha,
            beta_theta=args.beta_beta,
            kappa=args.beta_kappa,
            rng=rng,
        )
        Xn, _, meta = normalize_scores(X, dataset_key="synthetic", theta=theta, synthetic_ticks=10)
        return Xn, "Beta", meta
    raise ValueError(f"Unknown dataset: {name}")


def utility_vector(X: np.ndarray) -> np.ndarray:
    return np.nanmean(X, axis=1)


def row_interval_and_mean(row: np.ndarray, method: str) -> Tuple[Tuple[float, float], float]:
    vals = row[~np.isnan(row)].astype(float)
    mu = float(np.mean(vals))
    if method == "leave_one_out":
        if len(vals) <= 1:
            return (mu, mu), mu
        loo = []
        for j in range(len(vals)):
            loo.append(float(np.mean(np.delete(vals, j))))
        return (float(min(loo)), float(max(loo))), mu
    if method == "gaussian_ci":
        se = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        return (float(mu - 1.96 * se), float(mu + 1.96 * se)), mu
    if method == "minmax":
        return (float(np.min(vals)), float(np.max(vals))), mu
    raise ValueError(f"Unknown interval method: {method}")


def best_single_review_edit_for_row(
    row: np.ndarray,
    tick: float,
    method: str,
) -> Tuple[np.ndarray, Dict]:
    """Pick one-review +/-tick edit that maximizes interval+mean shift for this row."""
    base_intv, base_mu = row_interval_and_mean(row, method=method)
    obs = np.where(~np.isnan(row))[0]
    best = None
    best_row = None
    for j in obs:
        x0 = float(row[j])
        for s in (-1.0, 1.0):
            x1 = float(np.clip(x0 + s * tick, 0.0, 1.0))
            d_review = abs(x1 - x0)
            if d_review <= 0:
                continue
            rp = row.copy()
            rp[j] = x1
            intv_p, mu_p = row_interval_and_mean(rp, method=method)
            # Score the induced interval-model shift.
            shift_score = (
                abs(intv_p[0] - base_intv[0])
                + abs(mu_p - base_mu)
                + abs(intv_p[1] - base_intv[1])
            )
            rec = {
                "review_col": int(j),
                "direction": int(s),
                "x_before": x0,
                "x_after": x1,
                "review_delta": float(d_review),
                "interval_before_lo": float(base_intv[0]),
                "interval_before_hi": float(base_intv[1]),
                "interval_after_lo": float(intv_p[0]),
                "interval_after_hi": float(intv_p[1]),
                "mean_before": float(base_mu),
                "mean_after": float(mu_p),
                "interval_shift_score": float(shift_score),
            }
            if best is None or rec["interval_shift_score"] > best["interval_shift_score"]:
                best = rec
                best_row = rp
    if best is None:
        return row.copy(), {
            "review_col": -1,
            "direction": 0,
            "x_before": np.nan,
            "x_after": np.nan,
            "review_delta": 0.0,
            "interval_before_lo": float(base_intv[0]),
            "interval_before_hi": float(base_intv[1]),
            "interval_after_lo": float(base_intv[0]),
            "interval_after_hi": float(base_intv[1]),
            "mean_before": float(base_mu),
            "mean_after": float(base_mu),
            "interval_shift_score": 0.0,
        }
    return best_row, best


def baseline_probs(
    X: np.ndarray,
    intervals: List[Tuple[float, float]],
    x_mean: np.ndarray,
    k: int,
    m_band: int,
    include_merit: bool,
) -> Dict[str, np.ndarray]:
    p_swiss = swiss_nsf_intervals(intervals, list(x_mean), k)
    p_thresh = randomize_above_threshold(X, k=k, m=m_band)
    out = {
        "Swiss NSF": np.asarray(p_swiss, dtype=float),
        "Randomized Threshold": np.asarray(p_thresh, dtype=float),
    }
    if include_merit:
        out["MERIT"] = _run_merit(intervals, k)
    return out


def candidate_items(v: np.ndarray, k: int, window: int) -> List[int]:
    order = np.argsort(-v)
    n = len(v)
    lo = max(0, (k - 1) - window)
    hi = min(n, (k - 1) + window + 2)  # include k and k+1 neighborhood
    return [int(i) for i in order[lo:hi]]


def parse_mechanisms(spec: str) -> List[str]:
    allowed = {"MERIT", "Swiss NSF", "Randomized Threshold"}
    out = [x.strip() for x in spec.split(",") if x.strip()]
    for mech in out:
        if mech not in allowed:
            raise ValueError(f"Unknown mechanism '{mech}'. Allowed: {sorted(allowed)}")
    return out


def write_baseline_table_tex(summary: pd.DataFrame, out_tex: str) -> None:
    """Write LaTeX table for baseline local sensitivity summary."""
    mech_map = {
        "MERIT": "MERIT",
        "Swiss NSF": "Swiss NSF",
        "Randomized Threshold": "Threshold",
    }
    dataset_order = ["ICLR", "NeurIPS", "Swiss NSF", "Beta"]
    k_order = ["k10pct", "k33pct", "k50pct"]
    k_label = {"k10pct": "10\\%", "k33pct": "33\\%", "k50pct": "50\\%"}

    rows = []
    for d in dataset_order:
        for k_name in k_order:
            block = summary[(summary["dataset"] == d) & (summary["k_name"] == k_name)]
            row = {"dataset": d, "k": k_label[k_name]}
            for src, dst in mech_map.items():
                t = block[block["mechanism"] == src]
                if len(t):
                    row[f"{dst}_l1"] = float(t["l1_prob_change"].iloc[0])
                    row[f"{dst}_sens"] = float(t["local_sensitivity"].iloc[0])
                else:
                    row[f"{dst}_l1"] = np.nan
                    row[f"{dst}_sens"] = np.nan
            rows.append(row)

    def fmt(x: float) -> str:
        return "-" if pd.isna(x) else f"{x:.3f}"

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r" & & \multicolumn{2}{c}{MERIT} & \multicolumn{2}{c}{Swiss NSF} & \multicolumn{2}{c}{Threshold} \\",
        r"Dataset & $k$ & $\Delta p_{L1}$ & Sens. & $\Delta p_{L1}$ & Sens. & $\Delta p_{L1}$ & Sens. \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['dataset']} & {r['k']} & "
            f"{fmt(r['MERIT_l1'])} & {fmt(r['MERIT_sens'])} & "
            f"{fmt(r['Swiss NSF_l1'])} & {fmt(r['Swiss NSF_sens'])} & "
            f"{fmt(r['Threshold_l1'])} & {fmt(r['Threshold_sens'])} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Baseline local sensitivity under a single-review one-tick perturbation, using the edit that maximally shifts the perturbed item's LOO interval representation. $\Delta p_{L1}$ is the induced L1 change in marginal selection probabilities; Sens. is local sensitivity ($\Delta p_{L1}$/input perturbation magnitude).}",
        r"\label{tab:baseline_local_sensitivity}",
        r"\end{table}",
    ]
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run(args) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    datasets = ["beta", "neurips", "iclr", "swissnsf"]
    k_names = [x.strip() for x in args.k_names.split(",") if x.strip()]
    requested_mechanisms = parse_mechanisms(args.mechanisms)
    merit_requested = "MERIT" in requested_mechanisms
    merit_available = merit_requested
    if merit_requested:
        try:
            _ = _run_merit([(1.0, 1.0), (0.0, 0.0)], 1)
        except Exception as exc:
            merit_available = False
            print(f"[warn] MERIT unavailable ({type(exc).__name__}: {exc}). Skipping MERIT.")

    for i, dname in enumerate(datasets):
        rng = np.random.default_rng(args.seed + 1000 * (i + 1))
        X, dlabel, meta = load_dataset(dname, args, rng=rng)
        tick = float(meta["normalized_tick_size"])
        n = X.shape[0]
        v = utility_vector(X)
        rmin = r_min(X)
        l_ref = 1.0 / float(rmin)
        intervals, x_mean = reviews_to_intervals(X, method=args.interval_method)
        intervals = [tuple(map(float, itv)) for itv in intervals]
        x_mean = np.asarray(x_mean, dtype=float)

        for k_name in k_names:
            k = k_from_name(n, k_name)
            m_band = max(1, int(round(args.threshold_band_frac * k)))
            m_band = min(m_band, k, n - k) if (n - k) > 0 else 1

            p0 = baseline_probs(X, intervals, x_mean, k=k, m_band=m_band, include_merit=merit_available)
            base_cand = candidate_items(v, k=k, window=args.candidate_window)
            swiss_window = args.swiss_candidate_window
            swiss_cand = (
                list(range(n))
                if swiss_window < 0
                else candidate_items(v, k=k, window=swiss_window)
            )
            threshold_window = args.threshold_candidate_window
            threshold_cand = (
                list(range(n))
                if threshold_window < 0
                else candidate_items(v, k=k, window=threshold_window)
            )
            mech_order = [m for m in requested_mechanisms if (m != "MERIT" or merit_available)]
            for mech in mech_order:
                best = None
                if mech == "Swiss NSF":
                    cand = swiss_cand
                elif mech == "Randomized Threshold":
                    cand = threshold_cand
                else:
                    cand = base_cand
                for idx in cand:
                    # Select the single-review edit that maximally shifts this row's interval model.
                    row_p, edit = best_single_review_edit_for_row(
                        X[idx, :], tick=tick, method=args.interval_method
                    )
                    d_in = float(edit["review_delta"])
                    if d_in <= 0:
                        continue
                    Xp = X.copy()
                    Xp[idx, :] = row_p
                    intv_p, mu_p = row_interval_and_mean(row_p, method=args.interval_method)

                    if mech in ("MERIT", "Swiss NSF"):
                        intervals_p = list(intervals)
                        intervals_p[idx] = (float(intv_p[0]), float(intv_p[1]))
                        x_p = x_mean.copy()
                        x_p[idx] = float(mu_p)
                        if mech == "MERIT":
                            p1 = _run_merit(intervals_p, k)
                        else:
                            p1 = np.asarray(swiss_nsf_intervals(intervals_p, list(x_p), k), dtype=float)
                    else:
                        p1 = randomize_above_threshold(Xp, k=k, m=m_band)

                    l1 = float(np.abs(np.asarray(p1) - p0[mech]).sum())
                    sens = l1 / d_in
                    rec = {
                        "dataset": dlabel,
                        "k_name": k_name,
                        "k": int(k),
                        "n": int(n),
                        "r_min": int(rmin),
                        "L_ref": float(l_ref),
                        "mechanism": mech,
                        "item_idx": int(idx),
                        "review_col": int(edit["review_col"]),
                        "direction": int(edit["direction"]),
                        "tick": float(tick),
                        "input_delta": float(d_in),
                        "l1_prob_change": float(l1),
                        "local_sensitivity": float(sens),
                        "interval_shift_score": float(edit["interval_shift_score"]),
                    }
                    if best is None or rec["local_sensitivity"] > best["local_sensitivity"]:
                        best = rec
                if best is not None:
                    rows.append(best)

    df = pd.DataFrame(rows)
    if df.empty:
        return df, pd.DataFrame(columns=["dataset", "k_name", "k", "n", "r_min", "L_ref", "mechanism", "local_sensitivity"])
    summary = (
        df.groupby(["dataset", "k_name", "k", "n", "r_min", "L_ref", "mechanism"], as_index=False)
        .agg(
            local_sensitivity=("local_sensitivity", "max"),
            input_delta=("input_delta", "first"),
            l1_prob_change=("l1_prob_change", "first"),
            item_idx=("item_idx", "first"),
            review_col=("review_col", "first"),
            direction=("direction", "first"),
            interval_shift_score=("interval_shift_score", "first"),
        )
    )
    return df, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deterministic worst-case local sensitivity for baseline mechanisms")
    parser.add_argument("--k_names", default="k10pct,k33pct,k50pct")
    parser.add_argument("--interval_method", default="leave_one_out", choices=["leave_one_out", "gaussian_ci", "minmax"])
    parser.add_argument("--candidate_window", type=int, default=0)
    parser.add_argument("--swiss_candidate_window", type=int, default=50)
    parser.add_argument("--threshold_candidate_window", type=int, default=0)
    parser.add_argument("--threshold_band_frac", type=float, default=0.10)
    parser.add_argument(
        "--mechanisms",
        default="MERIT,Swiss NSF,Randomized Threshold",
        help="Comma-separated subset of: MERIT,Swiss NSF,Randomized Threshold",
    )
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

    df, summary = run(args)
    out_csv = os.path.join(args.output_dir, "baseline_local_sensitivity.csv")
    out_summary = os.path.join(args.output_dir, "baseline_local_sensitivity_summary.csv")
    out_tex = os.path.join(args.fig_dir, "baseline_local_sensitivity_table.tex")
    df.to_csv(out_csv, index=False)
    summary.to_csv(out_summary, index=False)
    if not summary.empty:
        write_baseline_table_tex(summary, out_tex=out_tex)
        print(f"Saved {out_tex}")
    print(f"Saved {out_csv}")
    print(f"Saved {out_summary}")
