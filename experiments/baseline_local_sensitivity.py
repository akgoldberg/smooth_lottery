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
import math
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
from mechanisms import (
    linear_lottery_mechanism,
    randomize_above_threshold,
    reviews_to_intervals,
    softmax_mechanism,
)
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
    L_smooth: float,
    softmax_samples: int,
    softmax_seed: int,
    include_merit: bool,
    include_linear: bool,
    include_softmax: bool,
) -> Dict[str, np.ndarray]:
    p_swiss = swiss_nsf_intervals(intervals, list(x_mean), k)
    p_thresh = randomize_above_threshold(X, k=k, m=m_band)
    out = {
        "Swiss NSF": np.asarray(p_swiss, dtype=float),
        "Randomized Threshold": np.asarray(p_thresh, dtype=float),
    }
    if include_merit:
        out["MERIT"] = _run_merit(intervals, k)
    if include_linear:
        out["Linear Lottery"] = linear_lottery_mechanism(X, k=k, L=L_smooth)
    if include_softmax:
        out["Softmax"] = softmax_mechanism(
            X,
            k=k,
            L=L_smooth,
            n_samples=softmax_samples,
            rng=np.random.default_rng(softmax_seed),
        )
    return out


def candidate_items(v: np.ndarray, k: int, window: int) -> List[int]:
    order = np.argsort(-v)
    n = len(v)
    lo = max(0, (k - 1) - window)
    hi = min(n, (k - 1) + window + 2)  # include k and k+1 neighborhood
    return [int(i) for i in order[lo:hi]]


def parse_mechanisms(spec: str) -> List[str]:
    allowed = {"MERIT", "Swiss NSF", "Randomized Threshold", "Linear Lottery", "Softmax"}
    out = [x.strip() for x in spec.split(",") if x.strip()]
    for mech in out:
        if mech not in allowed:
            raise ValueError(f"Unknown mechanism '{mech}'. Allowed: {sorted(allowed)}")
    return out


def expected_regret(v: np.ndarray, p: np.ndarray, k: int) -> float:
    top_sum = float(np.sort(v)[-k:].sum())
    return top_sum - float(np.dot(v, p))


def _softmax_tau_from_L(L: float, D_v: float, k: int) -> float:
    if k == 1:
        return D_v / (2.0 * L)
    return 2.0 * D_v / (math.e * L)


def _softmax_paired_probs_from_utils(
    v0: np.ndarray,
    v1: np.ndarray,
    k: int,
    L: float,
    D_v: float,
    samples: int,
    seed: int,
    reps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate softmax marginals for (v0, v1) using shared Gumbel draws."""
    n = len(v0)
    tau = _softmax_tau_from_L(L=L, D_v=D_v, k=k)
    p0_acc = np.zeros(n, dtype=float)
    p1_acc = np.zeros(n, dtype=float)
    reps = max(1, int(reps))
    for rep in range(reps):
        rng = np.random.default_rng(seed + 1009 * rep)
        g = rng.gumbel(loc=0.0, scale=tau, size=(samples, n))
        y0 = v0[None, :] + g
        y1 = v1[None, :] + g
        idx0 = np.argpartition(y0, kth=n - k, axis=1)[:, -k:]
        idx1 = np.argpartition(y1, kth=n - k, axis=1)[:, -k:]
        c0 = np.bincount(idx0.ravel(), minlength=n).astype(float)
        c1 = np.bincount(idx1.ravel(), minlength=n).astype(float)
        p0 = c0 / float(samples)
        p1 = c1 / float(samples)
        p0 = p0 / p0.sum() * k
        p1 = p1 / p1.sum() * k
        p0_acc += p0
        p1_acc += p1
    return p0_acc / reps, p1_acc / reps


def _write_table_block(
    summary: pd.DataFrame,
    mech_map: List[Tuple[str, str]],
    caption: str,
    label: str,
    out_tex: str,
) -> None:
    dataset_order = ["ICLR", "NeurIPS", "Swiss NSF", "Beta"]
    k_order = ["k10pct", "k33pct", "k50pct"]
    k_label = {"k10pct": "10\\%", "k33pct": "33\\%", "k50pct": "50\\%"}

    rows = []
    for d in dataset_order:
        for k_name in k_order:
            block = summary[(summary["dataset"] == d) & (summary["k_name"] == k_name)]
            row = {"dataset": d, "k": k_label[k_name]}
            for src, dst in mech_map:
                t = block[block["mechanism"] == src]
                if len(t):
                    row[f"{dst}_total_dp"] = float(t["l1_prob_change"].iloc[0])
                    row[f"{dst}_max_dp"] = float(t["max_prob_change"].iloc[0])
                    row[f"{dst}_sens"] = float(t["local_sensitivity"].iloc[0])
                    if "expected_regret_per_k" in t.columns:
                        row[f"{dst}_regret_per_k"] = float(t["expected_regret_per_k"].iloc[0])
                    elif "expected_regret" in t.columns and "k" in t.columns:
                        kval = float(t["k"].iloc[0])
                        row[f"{dst}_regret_per_k"] = float(t["expected_regret"].iloc[0]) / kval if kval > 0 else np.nan
                    else:
                        row[f"{dst}_regret_per_k"] = np.nan
                else:
                    row[f"{dst}_total_dp"] = np.nan
                    row[f"{dst}_max_dp"] = np.nan
                    row[f"{dst}_sens"] = np.nan
                    row[f"{dst}_regret_per_k"] = np.nan
            rows.append(row)

    def fmt_dp(x: float) -> str:
        return "-" if pd.isna(x) else f"{x:.3f}"

    smooth_only = all(label in {"Linear Lottery", "Softmax"} for _, label in mech_map)

    def fmt_s(x: float) -> str:
        if pd.isna(x):
            return "-"
        return f"{x:.2f}" if smooth_only else f"{x:.1f}"

    def fmt_rk(x: float) -> str:
        return "-" if pd.isna(x) else f"{x:.3f}"

    colspec = "ll|" + "|".join(["rrrr" for _ in mech_map])
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{" + colspec + "}",
        r"\toprule",
        r" & & " + " & ".join([rf"\multicolumn{{4}}{{c}}{{{label}}}" for _, label in mech_map]) + r" \\",
        r"Dataset & $k$ & " + " & ".join([r"Total $\Delta p$ & Max $\Delta p$ & Smoothness & Regret$/k$" for _ in mech_map]) + r" \\",
        r"\midrule",
    ]
    for d_idx, d in enumerate(dataset_order):
        d_rows = [r for r in rows if r["dataset"] == d]
        for r in d_rows:
            lines.append(
                f"{r['dataset']} & {r['k']} & "
                + " & ".join(
                    [
                        f"{fmt_dp(r[f'{label}_total_dp'])} & {fmt_dp(r[f'{label}_max_dp'])} & {fmt_s(r[f'{label}_sens'])}"
                        f" & {fmt_rk(r[f'{label}_regret_per_k'])}"
                        for _, label in mech_map
                    ]
                )
                + r" \\"
            )
        if d_idx < len(dataset_order) - 1:
            lines.append(r"\midrule")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\end{table}",
    ]
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_baseline_tables_tex(summary: pd.DataFrame, fig_dir: str) -> List[str]:
    """Write separate LaTeX tables for existing baselines and smooth mechanisms."""
    base_mechs = [
        ("MERIT", "MERIT"),
        ("Swiss NSF", "Swiss NSF"),
        ("Randomized Threshold", "Threshold"),
    ]
    smooth_mechs = [
        ("Linear Lottery", "Linear Lottery"),
        ("Softmax", "Softmax"),
    ]
    caption_common = (
        "Local sensitivity under a single-review one-tick perturbation, using the edit that "
        "maximally shifts the perturbed item's LOO interval representation. Total $\\Delta p$ "
        "is the L1 change in the marginal probability vector, Max $\\Delta p$ is the maximum "
        "coordinate change, Smoothness is local sensitivity ($\\Delta p_{L1}$/input perturbation magnitude), "
        "and Regret$/k$ is expected regret normalized by $k$ on the unperturbed input."
    )
    out1 = os.path.join(fig_dir, "baseline_local_sensitivity_table_existing.tex")
    out2 = os.path.join(fig_dir, "baseline_local_sensitivity_table_smooth.tex")
    _write_table_block(
        summary=summary,
        mech_map=base_mechs,
        caption=f"Existing baseline partial lotteries. {caption_common}",
        label="tab:baseline_local_sensitivity_existing",
        out_tex=out1,
    )
    _write_table_block(
        summary=summary,
        mech_map=smooth_mechs,
        caption=f"Smooth mechanisms at $L=1/r$. {caption_common}",
        label="tab:baseline_local_sensitivity_smooth",
        out_tex=out2,
    )
    return [out1, out2]


def run(args) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    datasets = ["beta", "neurips", "iclr", "swissnsf"]
    k_names = [x.strip() for x in args.k_names.split(",") if x.strip()]
    requested_mechanisms = parse_mechanisms(args.mechanisms)
    merit_requested = "MERIT" in requested_mechanisms
    linear_requested = "Linear Lottery" in requested_mechanisms
    softmax_requested = "Softmax" in requested_mechanisms
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

            p0 = baseline_probs(
                X,
                intervals,
                x_mean,
                k=k,
                m_band=m_band,
                L_smooth=l_ref,
                softmax_samples=args.softmax_samples,
                softmax_seed=args.seed + 100_000 + 1000 * i + k,
                include_merit=merit_available,
                include_linear=linear_requested,
                include_softmax=False,
            )
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
                    elif mech == "Randomized Threshold":
                        p1 = randomize_above_threshold(Xp, k=k, m=m_band)
                    elif mech == "Linear Lottery":
                        p1 = linear_lottery_mechanism(Xp, k=k, L=l_ref)
                    elif mech == "Softmax":
                        v0 = np.nanmean(X, axis=1)
                        v1 = np.nanmean(Xp, axis=1)
                        p0_soft, p1 = _softmax_paired_probs_from_utils(
                            v0=v0,
                            v1=v1,
                            k=k,
                            L=l_ref,
                            D_v=1.0 / float(rmin),
                            samples=args.softmax_samples,
                            seed=args.seed + 200_000 + 10_000 * i + 100 * k + int(idx),
                            reps=args.softmax_reps,
                        )
                        p0_mech = p0_soft
                    else:
                        raise ValueError(f"Unknown mechanism in loop: {mech}")

                    if mech == "Softmax":
                        base_p = np.asarray(p0_mech)
                    else:
                        base_p = np.asarray(p0[mech])
                    l1 = float(np.abs(np.asarray(p1) - base_p).sum())
                    max_dp = float(np.max(np.abs(np.asarray(p1) - base_p)))
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
                        "max_prob_change": float(max_dp),
                        "local_sensitivity": float(sens),
                        "expected_regret": float(expected_regret(v, base_p, k)),
                        "expected_regret_per_k": float(expected_regret(v, base_p, k) / float(k)),
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
            max_prob_change=("max_prob_change", "first"),
            expected_regret=("expected_regret", "first"),
            expected_regret_per_k=("expected_regret_per_k", "first"),
            item_idx=("item_idx", "first"),
            review_col=("review_col", "first"),
            direction=("direction", "first"),
            interval_shift_score=("interval_shift_score", "first"),
        )
    )
    return df, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Worst-case local sensitivity for existing baseline partial-lottery mechanisms")
    parser.add_argument("--k_names", default="k10pct,k33pct,k50pct")
    parser.add_argument("--interval_method", default="leave_one_out", choices=["leave_one_out", "gaussian_ci", "minmax"])
    parser.add_argument("--candidate_window", type=int, default=0)
    parser.add_argument("--swiss_candidate_window", type=int, default=50)
    parser.add_argument("--threshold_candidate_window", type=int, default=0)
    parser.add_argument("--threshold_band_frac", type=float, default=0.10)
    parser.add_argument(
        "--mechanisms",
        default="MERIT,Swiss NSF,Randomized Threshold,Linear Lottery,Softmax",
        help="Comma-separated subset of: MERIT,Swiss NSF,Randomized Threshold,Linear Lottery,Softmax",
    )
    parser.add_argument("--softmax_samples", type=int, default=20000)
    parser.add_argument("--softmax_reps", type=int, default=5)
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
    df.to_csv(out_csv, index=False)
    summary.to_csv(out_summary, index=False)
    if not summary.empty:
        out_texs = write_baseline_tables_tex(summary, fig_dir=args.fig_dir)
        for out_tex in out_texs:
            print(f"Saved {out_tex}")
    print(f"Saved {out_csv}")
    print(f"Saved {out_summary}")
