#!/usr/bin/env python3
"""Local smoothness under single-review perturbations at ranks k and k+1.

Runs one sample for each dataset (Beta, NeurIPS, ICLR, Swiss NSF), sweeps
k settings and target L values, and computes local smoothness for Linear
Lottery and Softmax using one-review one-tick perturbations.
"""

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

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
from mechanisms import linear_lottery_mechanism, softmax_mechanism
from plot_results import plot_local_smoothness_2x1
from utils import drop_low_review_outliers, k_from_name, normalize_scores


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


def item_at_rank(v: np.ndarray, rank_1_based: int) -> Optional[int]:
    n = len(v)
    if rank_1_based < 1 or rank_1_based > n:
        return None
    order = np.argsort(-v)
    return int(order[rank_1_based - 1])


def first_observed_review_idx(X: np.ndarray, item_idx: int) -> Optional[int]:
    cols = np.where(~np.isnan(X[item_idx]))[0]
    if len(cols) == 0:
        return None
    return int(cols[0])


def candidate_perturbations(X: np.ndarray, item_idx: int, tick: float) -> List[Tuple[np.ndarray, float, int]]:
    j = first_observed_review_idx(X, item_idx)
    if j is None:
        return []
    x0 = float(X[item_idx, j])
    out = []
    for s in (-1.0, 1.0):
        x1 = float(np.clip(x0 + s * tick, 0.0, 1.0))
        d = abs(x1 - x0)
        if d <= 0:
            continue
        Xp = X.copy()
        Xp[item_idx, j] = x1
        out.append((Xp, d, j))
    return out


def mechanism_probs(X: np.ndarray, k: int, L: float, softmax_samples: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    p_lin = linear_lottery_mechanism(X, k, L)
    p_soft = softmax_mechanism(X, k, L, n_samples=softmax_samples, rng=rng)
    return {"linear": p_lin, "softmax": p_soft}


def _estimate_softmax_l1_change(
    X: np.ndarray,
    Xp: np.ndarray,
    k: int,
    L: float,
    softmax_samples: int,
    reps: int,
    seed: int,
) -> float:
    vals = []
    for rep in range(reps):
        rep_seed = seed + 104729 * (rep + 1)
        p_base = mechanism_probs(
            X, k, L, softmax_samples=softmax_samples, rng=np.random.default_rng(rep_seed)
        )["softmax"]
        p_pert = mechanism_probs(
            Xp, k, L, softmax_samples=softmax_samples, rng=np.random.default_rng(rep_seed)
        )["softmax"]
        vals.append(float(np.abs(p_pert - p_base).sum()))
    return float(np.mean(vals))


def evaluate_single_item(
    X: np.ndarray,
    dataset: str,
    rank_label: str,
    item_idx: int,
    k: int,
    k_name: str,
    L: float,
    tick: float,
    softmax_search_samples: int,
    softmax_search_reps: int,
    softmax_final_samples: int,
    softmax_final_reps: int,
    seed: int,
) -> List[Dict]:
    out = []
    p_base_linear = mechanism_probs(
        X, k, L, softmax_samples=softmax_search_samples, rng=np.random.default_rng(seed)
    )["linear"]

    v = utility_vector(X)
    cands = candidate_perturbations(X, item_idx=item_idx, tick=tick)
    if not cands:
        return out

    for mech in ("linear", "softmax"):
        best = None
        for cand_id, (Xp, l11_delta, review_col) in enumerate(cands):
            if mech == "linear":
                p_pert = mechanism_probs(
                    Xp, k, L, softmax_samples=softmax_search_samples, rng=np.random.default_rng(seed)
                )["linear"]
                l1_prob = float(np.abs(p_pert - p_base_linear).sum())
            else:
                search_seed = seed + 1009 * (cand_id + 1)
                l1_prob = _estimate_softmax_l1_change(
                    X=X,
                    Xp=Xp,
                    k=k,
                    L=L,
                    softmax_samples=softmax_search_samples,
                    reps=softmax_search_reps,
                    seed=search_seed,
                )

            local = l1_prob / l11_delta if l11_delta > 0 else np.nan
            v_pert = utility_vector(Xp)
            rec = {
                "dataset": dataset,
                "k_name": k_name,
                "rank_target": rank_label,
                "mechanism": mech,
                "item_idx": int(item_idx),
                "review_col": int(review_col),
                "k": int(k),
                "L": float(L),
                "tick": float(tick),
                "l11_delta": float(l11_delta),
                "utility_before": float(v[item_idx]),
                "utility_after": float(v_pert[item_idx]),
                "utility_delta_abs": float(abs(v_pert[item_idx] - v[item_idx])),
                "l1_prob_change": float(l1_prob),
                "local_smoothness": float(local),
            }
            if best is None or rec["local_smoothness"] > best["local_smoothness"]:
                best = rec
        if best is not None:
            # Re-estimate the selected softmax candidate with fresh randomness.
            if mech == "softmax":
                # Find the selected candidate again.
                selected_cand = None
                for cand_id, (Xp, l11_delta, review_col) in enumerate(cands):
                    if int(review_col) != int(best["review_col"]):
                        continue
                    util_after = float(utility_vector(Xp)[item_idx])
                    if abs(util_after - float(best["utility_after"])) < 1e-12:
                        selected_cand = (cand_id, Xp, l11_delta)
                        break
                if selected_cand is not None:
                    cand_id, Xp_best, l11_delta_best = selected_cand
                    final_seed = seed + 10_000_019 + 1009 * (cand_id + 1)
                    final_l1 = _estimate_softmax_l1_change(
                        X=X,
                        Xp=Xp_best,
                        k=k,
                        L=L,
                        softmax_samples=softmax_final_samples,
                        reps=softmax_final_reps,
                        seed=final_seed,
                    )
                    best["l1_prob_change"] = float(final_l1)
                    best["local_smoothness"] = float(final_l1 / l11_delta_best) if l11_delta_best > 0 else np.nan
                    best["softmax_search_samples"] = int(softmax_search_samples)
                    best["softmax_search_reps"] = int(softmax_search_reps)
                    best["softmax_final_samples"] = int(softmax_final_samples)
                    best["softmax_final_reps"] = int(softmax_final_reps)
                else:
                    best["softmax_search_samples"] = int(softmax_search_samples)
                    best["softmax_search_reps"] = int(softmax_search_reps)
                    best["softmax_final_samples"] = int(softmax_final_samples)
                    best["softmax_final_reps"] = int(softmax_final_reps)
            else:
                best["softmax_search_samples"] = 0
                best["softmax_search_reps"] = 0
                best["softmax_final_samples"] = 0
                best["softmax_final_reps"] = 0
            out.append(best)
    return out


def run(args) -> pd.DataFrame:
    datasets = ["beta", "neurips", "iclr", "swissnsf"]
    k_names = [x.strip() for x in args.k_names.split(",") if x.strip()]
    L_values = [float(x.strip()) for x in args.L_values.split(",") if x.strip()]

    rows = []
    for i, name in enumerate(datasets):
        rng = np.random.default_rng(args.seed + 1000 * (i + 1))
        X, label, meta = load_dataset(name, args, rng=rng)
        tick = float(meta["normalized_tick_size"])
        v = utility_vector(X)

        for k_name in k_names:
            n = X.shape[0]
            k = k_from_name(n, k_name)
            idx_k = item_at_rank(v, k)
            idx_k1 = item_at_rank(v, k + 1)
            targets = [("k", idx_k), ("k+1", idx_k1)]

            for L in L_values:
                for rank_label, idx in targets:
                    if idx is None:
                        continue
                    rows.extend(
                        evaluate_single_item(
                            X=X,
                            dataset=label,
                            rank_label=rank_label,
                            item_idx=idx,
                            k=k,
                            k_name=k_name,
                            L=L,
                            tick=tick,
                            softmax_search_samples=args.softmax_search_samples,
                            softmax_search_reps=args.softmax_search_reps,
                            softmax_final_samples=args.softmax_final_samples,
                            softmax_final_reps=args.softmax_final_reps,
                            seed=args.seed + 100000 * (i + 1) + 1000 * (k + 1) + int(round(100 * L)) + (0 if rank_label == "k" else 1),
                        )
                    )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local smoothness at k / k+1 single perturbations")
    parser.add_argument("--k_names", default="k10pct,k33pct,k50pct")
    parser.add_argument("--L_values", default="0.2,0.4,0.6,0.8,1.0")
    parser.add_argument("--softmax_search_samples", type=int, default=1500)
    parser.add_argument("--softmax_search_reps", type=int, default=2)
    parser.add_argument("--softmax_final_samples", type=int, default=8000)
    parser.add_argument("--softmax_final_reps", type=int, default=3)
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

    df = run(args)
    out_csv = os.path.join(args.output_dir, "local_smoothness_all.csv")
    out_summary = os.path.join(args.output_dir, "local_smoothness_all_summary.csv")
    df.to_csv(out_csv, index=False)

    if not df.empty:
        summary = (
            df.groupby(["dataset", "k_name", "L", "mechanism"], as_index=False)
            .agg(
                local_smoothness=("local_smoothness", "max"),
                l1_prob_change=("l1_prob_change", "max"),
                utility_delta_abs=("utility_delta_abs", "max"),
                k=("k", "first"),
            )
        )
        summary["ratio_to_L"] = summary["local_smoothness"] / summary["L"]
        summary.to_csv(out_summary, index=False)

        for k_name in sorted(summary["k_name"].unique()):
            sub = summary[summary["k_name"] == k_name].copy()
            out_pdf = os.path.join(args.fig_dir, f"local_smoothness_{k_name}.pdf")
            plot_local_smoothness_2x1(sub, k_name=k_name, out_pdf=out_pdf)
            print(f"Saved {out_pdf}")
    else:
        pd.DataFrame(
            columns=[
                "dataset", "k_name", "L", "mechanism", "local_smoothness",
                "l1_prob_change", "utility_delta_abs", "k", "ratio_to_L",
            ]
        ).to_csv(out_summary, index=False)

    print(f"Saved {out_csv}")
    print(f"Saved {out_summary}")
