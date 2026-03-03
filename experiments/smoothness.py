#!/usr/bin/env python3
"""
Global smoothness experiment for Linear Lottery and Softmax.

Purpose:
- Empirically estimate near-worst-case global smoothness constants.
- Compare empirical smoothness against target/theoretical smoothness levels.

Method:
- Uses threshold-family utility constructions parameterized by (n, k, B, eps).
- Searches over perturbation families to maximize local L1 probability change / input change.
- Linear Lottery is exact; Softmax uses shared-noise Monte Carlo with search and final re-estimation.

Main outputs:
- `experiments/results/global_smoothness_summary.csv`
- `experiments/results/epsilon_response_n100_k10_L1.csv`
- `experiments/figures/global_smoothness_*_empirical_vs_targetL.pdf`
- `experiments/figures/global_smoothness_*_ratio_vs_targetL.pdf`
- `experiments/figures/epsilon_response_n100_k10_L1.pdf`
"""

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_BASE = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, _BASE)
from smooth_lotteries import linear_lottery
from plot_results import plot_global_smoothness_2x1


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
FIGURES = ROOT / "experiments" / "figures"


def compute_k_values(n: int) -> List[int]:
    vals = [int(round(0.10 * n)), int(round((1.0 / 3.0) * n)), int(round(0.50 * n))]
    vals = [min(max(1, v), n - 1) for v in vals]
    # keep order but dedupe
    out = []
    for v in vals:
        if v not in out:
            out.append(v)
    return out


def clear_old(prefix: str = "global_smoothness_") -> None:
    for p in RESULTS.glob(f"{prefix}*"):
        p.unlink()
    for p in FIGURES.glob(f"{prefix}*"):
        p.unlink()


def _softmax_tau_from_bound(theorem_bound: float) -> float:
    return 2.0 / (math.e * theorem_bound)


def _paper_l_from_bound(theorem_bound: float, r: float = 5.0) -> float:
    return theorem_bound / r


def _build_threshold_family(
    n: int,
    k: int,
    B: float,
    eps: float,
    direction: int = 1,
) -> Tuple[np.ndarray, np.ndarray, int]:
    if not (1 <= k <= n):
        raise ValueError("Require 1 <= k <= n.")
    if not (0.0 <= B <= 1.0):
        raise ValueError("Require B in [0,1].")
    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or +1")
    if eps < 0:
        raise ValueError("Require eps >= 0.")
    B2 = B + direction * eps
    if B2 < 0.0 or B2 > 1.0:
        raise ValueError("Require perturbed value B + direction*eps in [0,1].")
    u = np.zeros(n, dtype=float)
    if k > 1:
        u[:k - 1] = 1.0
    target_idx = k - 1
    u[target_idx] = B
    u2 = u.copy()
    u2[target_idx] = B2
    return u, u2, target_idx


def _estimate_softmax_marginals_shared_gumbels(
    u: np.ndarray,
    u2: np.ndarray,
    k: int,
    tau: float,
    samples: int,
    seed: int,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = u.shape[0]
    counts1 = np.zeros(n, dtype=float)
    counts2 = np.zeros(n, dtype=float)
    done = 0
    batch_size = max(1, int(batch_size))
    while done < samples:
        m = min(batch_size, samples - done)
        g = rng.gumbel(loc=0.0, scale=tau, size=(m, n))
        y1 = u[None, :] + g
        y2 = u2[None, :] + g
        idx1 = np.argpartition(y1, kth=n-k, axis=1)[:, -k:]
        idx2 = np.argpartition(y2, kth=n-k, axis=1)[:, -k:]
        counts1 += np.bincount(idx1.ravel(), minlength=n)
        counts2 += np.bincount(idx2.ravel(), minlength=n)
        done += m
    return counts1 / samples, counts2 / samples


def _run_softmax_trial(
    n: int,
    k: int,
    theorem_bound: float,
    B: float,
    eps: float,
    samples: int,
    seed: int,
    batch_size: int,
    direction: int = 1,
) -> Dict:
    tau = _softmax_tau_from_bound(theorem_bound)
    u, u2, target_idx = _build_threshold_family(n=n, k=k, B=B, eps=eps, direction=direction)
    p1, p2 = _estimate_softmax_marginals_shared_gumbels(
        u=u, u2=u2, k=k, tau=tau, samples=samples, seed=seed, batch_size=batch_size
    )
    l1_change = float(np.abs(p2 - p1).sum())
    quotient = l1_change / eps if eps > 0 else float("nan")
    ratio = quotient / theorem_bound if theorem_bound > 0 else float("nan")
    return {
        "B": float(B),
        "eps": float(eps),
        "direction": int(direction),
        "mc_samples": int(samples),
        "quotient": float(quotient),
        "ratio_to_bound": float(ratio),
        "target_prob_base": float(p1[target_idx]),
        "target_prob_pert": float(p2[target_idx]),
    }


def _run_linear_trial(
    n: int,
    k: int,
    theorem_bound: float,
    B: float,
    eps: float,
    direction: int = 1,
) -> Dict:
    w = theorem_bound / 2.0
    u, u2, target_idx = _build_threshold_family(n=n, k=k, B=B, eps=eps, direction=direction)
    p1, _ = linear_lottery(u, k=k, w=w)
    p2, _ = linear_lottery(u2, k=k, w=w)
    l1_change = float(np.abs(p2 - p1).sum())
    quotient = l1_change / eps if eps > 0 else float("nan")
    ratio = quotient / theorem_bound if theorem_bound > 0 else float("nan")
    return {
        "B": float(B),
        "eps": float(eps),
        "direction": int(direction),
        "mc_samples": 0,
        "quotient": float(quotient),
        "ratio_to_bound": float(ratio),
        "target_prob_base": float(p1[target_idx]),
        "target_prob_pert": float(p2[target_idx]),
    }


def run_softmax_block(
    n: int,
    ks: List[int],
    theorem_bounds: List[float],
    search_samples: int,
    final_samples: int,
    batch_size: int,
    seed: int,
) -> List[Dict]:
    B_grid = [0.50, 0.75, 0.95, 1.00]
    eps_grid = [0.04, 0.02, 0.01]
    rows: List[Dict] = []
    for k in ks:
        for tb in theorem_bounds:
            tau = _softmax_tau_from_bound(tb)
            best = None
            t_idx = 0
            for B in B_grid:
                for eps in eps_grid:
                    for direction in (1, -1):
                        if not (0.0 <= B + direction * eps <= 1.0):
                            continue
                        t_seed = seed + 100003 * t_idx + 17 * k + int(round(1000 * tau))
                        t_idx += 1
                        trial = _run_softmax_trial(
                            n=n,
                            k=k,
                            theorem_bound=tb,
                            B=B,
                            eps=eps,
                            samples=search_samples,
                            seed=t_seed,
                            batch_size=batch_size,
                            direction=direction,
                        )
                        if best is None or trial["quotient"] > best["quotient"]:
                            best = trial
            assert best is not None
            final_seed = seed + 10000019 + 17 * k + int(round(1000 * tau))
            final = _run_softmax_trial(
                n=n, k=k, theorem_bound=tb, B=best["B"], eps=best["eps"],
                samples=final_samples, seed=final_seed, batch_size=batch_size,
                direction=best["direction"],
            )
            rows.append(
                {
                    "mechanism": "softmax",
                    "n": n,
                    "k": k,
                    "theorem_bound": tb,
                    "L": _paper_l_from_bound(tb),
                    "B": final["B"],
                    "eps": final["eps"],
                    "direction": final["direction"],
                    "mc_samples": final["mc_samples"],
                    "quotient": final["quotient"],
                    "best_empirical_L": final["quotient"] / 5.0,
                    "ratio_empirical_to_targetL": final["ratio_to_bound"],
                }
            )
            print(
                f"[softmax] n={n:>4} k={k:>4} L={_paper_l_from_bound(tb):.3f} "
                f"ratio={final['ratio_to_bound']:.3f} B={final['B']:.3f} "
                f"eps={final['eps']:.4f} dir={final['direction']:+d}"
            )
    return rows


def run_linear_block(
    n: int,
    ks: List[int],
    theorem_bounds: List[float],
) -> List[Dict]:
    # Sparse on purpose (user request).
    B_grid = [0.10, 0.30, 0.50, 0.70, 0.90, 1.00]
    eps_grid = [0.001, 0.005, 0.01]
    rows: List[Dict] = []
    for k in ks:
        for tb in theorem_bounds:
            best = None
            for B in B_grid:
                for eps in eps_grid:
                    for direction in (1, -1):
                        if not (0.0 <= B + direction * eps <= 1.0):
                            continue
                        trial = _run_linear_trial(
                            n=n, k=k, theorem_bound=tb, B=B, eps=eps, direction=direction
                        )
                        if best is None or trial["quotient"] > best["quotient"]:
                            best = trial
            assert best is not None
            rows.append(
                {
                    "mechanism": "linear",
                    "n": n,
                    "k": k,
                    "theorem_bound": tb,
                    "L": _paper_l_from_bound(tb),
                    "B": best["B"],
                    "eps": best["eps"],
                    "direction": best["direction"],
                    "mc_samples": 0,
                    "quotient": best["quotient"],
                    "best_empirical_L": best["quotient"] / 5.0,
                    "ratio_empirical_to_targetL": best["ratio_to_bound"],
                }
            )
            print(
                f"[linear ] n={n:>4} k={k:>4} L={_paper_l_from_bound(tb):.3f} "
                f"ratio={best['ratio_to_bound']:.3f} B={best['B']:.3f} "
                f"eps={best['eps']:.4f} dir={best['direction']:+d}"
            )
    return rows


def write_summary(rows: List[Dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "mechanism", "n", "k", "theorem_bound", "L",
        "B", "eps", "direction", "mc_samples", "quotient",
        "best_empirical_L", "ratio_empirical_to_targetL",
    ]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _lookup_row(rows: List[Dict], mechanism: str, n: int, k: int, L: float) -> Dict:
    for row in rows:
        if (
            row["mechanism"] == mechanism
            and int(row["n"]) == int(n)
            and int(row["k"]) == int(k)
            and abs(float(row["L"]) - float(L)) < 1e-12
        ):
            return row
    raise ValueError(f"Missing row for mechanism={mechanism}, n={n}, k={k}, L={L}")


def _estimate_softmax_target_prob(
    u: np.ndarray,
    target_idx: int,
    k: int,
    tau: float,
    samples: int,
    seed: int,
    batch_size: int,
) -> float:
    rng = np.random.default_rng(seed)
    n = len(u)
    total = 0.0
    done = 0
    while done < samples:
        m = min(batch_size, samples - done)
        g = rng.gumbel(loc=0.0, scale=tau, size=(m, n))
        y = u[None, :] + g
        idx = np.argpartition(y, kth=n-k, axis=1)[:, -k:]
        total += np.count_nonzero(idx == target_idx)
        done += m
    return float(total / samples)


def _epsilon_response_panels(
    rows: List[Dict],
    seed: int,
    softmax_samples: int,
    batch_size: int,
) -> Tuple[Path, Path]:
    n = 100
    k = int(round(0.10 * n))
    L = 1.0
    eps_grid = np.arange(0.0, 1.0 + 1e-12, 0.05)
    target = k - 1

    row_lin = _lookup_row(rows, mechanism="linear", n=n, k=k, L=L)
    row_soft = _lookup_row(rows, mechanism="softmax", n=n, k=k, L=L)

    # Near-uniform: u_i = 0 and u_1 -> u_1 + eps
    near_lin = []
    near_soft = []
    for i, eps in enumerate(eps_grid):
        u0 = np.zeros(n, dtype=float)
        u1 = u0.copy()
        u1[0] = eps
        p0, _ = linear_lottery_smooth(u0, k=k, L=L, D_v=1.0)
        p1, _ = linear_lottery_smooth(u1, k=k, L=L, D_v=1.0)
        near_lin.append(float(p1[0] - p0[0]))

        tau = _softmax_tau_from_bound(float(row_soft["theorem_bound"]))
        s0 = _estimate_softmax_target_prob(
            u=u0,
            target_idx=0,
            k=k,
            tau=tau,
            samples=softmax_samples,
            seed=seed + 10_000 + i,
            batch_size=batch_size,
        )
        s1 = _estimate_softmax_target_prob(
            u=u1,
            target_idx=0,
            k=k,
            tau=tau,
            samples=softmax_samples,
            seed=seed + 20_000 + i,
            batch_size=batch_size,
        )
        near_soft.append(float(s1 - s0))

    # Worst-case family: u_1..u_{k-1}=1, u_k=B, rest=0; perturb u_k -> u_k + eps (clipped)
    worst_lin = []
    worst_soft = []
    for i, eps in enumerate(eps_grid):
        ub = np.zeros(n, dtype=float)
        ub[:k-1] = 1.0
        ub[target] = float(row_lin["B"])
        up = ub.copy()
        up[target] = float(np.clip(up[target] + eps, 0.0, 1.0))

        p0, _ = linear_lottery(ub, k=k, w=float(row_lin["theorem_bound"]) / 2.0)
        p1, _ = linear_lottery(up, k=k, w=float(row_lin["theorem_bound"]) / 2.0)
        worst_lin.append(float(p1[target] - p0[target]))

        tau = _softmax_tau_from_bound(float(row_soft["theorem_bound"]))
        sb = np.zeros(n, dtype=float)
        sb[:k-1] = 1.0
        sb[target] = float(row_soft["B"])
        sp = sb.copy()
        sp[target] = float(np.clip(sp[target] + eps, 0.0, 1.0))
        s0 = _estimate_softmax_target_prob(
            u=sb,
            target_idx=target,
            k=k,
            tau=tau,
            samples=softmax_samples,
            seed=seed + 30_000 + i,
            batch_size=batch_size,
        )
        s1 = _estimate_softmax_target_prob(
            u=sp,
            target_idx=target,
            k=k,
            tau=tau,
            samples=softmax_samples,
            seed=seed + 40_000 + i,
            batch_size=batch_size,
        )
        worst_soft.append(float(s1 - s0))

    # Top-k edge: u_1..u_k=1, rest=0; perturb u_k -> 1-eps
    edge_lin = []
    edge_soft = []
    for i, eps in enumerate(eps_grid):
        ub = np.zeros(n, dtype=float)
        ub[:k] = 1.0
        up = ub.copy()
        up[target] = float(np.clip(1.0 - eps, 0.0, 1.0))
        p0, _ = linear_lottery_smooth(ub, k=k, L=L, D_v=1.0)
        p1, _ = linear_lottery_smooth(up, k=k, L=L, D_v=1.0)
        edge_lin.append(float(p1[target] - p0[target]))

        tau = _softmax_tau_from_bound(float(row_soft["theorem_bound"]))
        s0 = _estimate_softmax_target_prob(
            u=ub,
            target_idx=target,
            k=k,
            tau=tau,
            samples=softmax_samples,
            seed=seed + 50_000 + i,
            batch_size=batch_size,
        )
        s1 = _estimate_softmax_target_prob(
            u=up,
            target_idx=target,
            k=k,
            tau=tau,
            samples=softmax_samples,
            seed=seed + 60_000 + i,
            batch_size=batch_size,
        )
        edge_soft.append(float(s1 - s0))

    out_csv = RESULTS / "epsilon_response_n100_k10_L1.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "epsilon",
            "near_uniform_delta_p1_linear",
            "near_uniform_delta_p1_softmax",
            "worst_case_delta_pk_linear",
            "worst_case_delta_pk_softmax",
            "edge_delta_pk_linear",
            "edge_delta_pk_softmax",
        ])
        for i, eps in enumerate(eps_grid):
            w.writerow([
                float(eps),
                float(near_lin[i]),
                float(near_soft[i]),
                float(worst_lin[i]),
                float(worst_soft[i]),
                float(edge_lin[i]),
                float(edge_soft[i]),
            ])

    out_pdf = FIGURES / "epsilon_response_n100_k10_L1.pdf"
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6), constrained_layout=True)
    style_lin = dict(color="#1f77b4", marker="o", linestyle="-", lw=2.2, ms=4)
    style_soft = dict(color="#ff7f0e", marker="s", linestyle="--", lw=2.2, ms=4)

    axes[0].plot(eps_grid, near_lin, **style_lin, label="Linear Lottery")
    axes[0].plot(eps_grid, near_soft, **style_soft, label="Softmax")
    axes[0].set_title("Near-Uniform")
    axes[0].set_xlabel(r"$\epsilon$")
    axes[0].set_ylabel(r"$\Delta p_1$")
    axes[0].grid(alpha=0.25)

    axes[1].plot(eps_grid, worst_lin, **style_lin)
    axes[1].plot(eps_grid, worst_soft, **style_soft)
    axes[1].set_title("Worst-Case Family")
    axes[1].set_xlabel(r"$\epsilon$")
    axes[1].set_ylabel(r"$\Delta p_k$")
    axes[1].grid(alpha=0.25)

    axes[2].plot(eps_grid, edge_lin, **style_lin)
    axes[2].plot(eps_grid, edge_soft, **style_soft)
    axes[2].set_title("Top-k Edge")
    axes[2].set_xlabel(r"$\epsilon$")
    axes[2].set_ylabel(r"$\Delta p_k$")
    axes[2].grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=True)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out_csv, out_pdf


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--skip-clean", action="store_true")
    p.add_argument("--softmax-search-samples", type=int, default=5000)
    p.add_argument("--softmax-final-samples", type=int, default=50000)
    p.add_argument("--softmax-batch-size", type=int, default=10000)
    p.add_argument("--epsilon-softmax-samples", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if not args.skip_clean:
        clear_old(prefix="global_smoothness_")

    theorem_bounds = [1.0, 2.0, 3.0, 4.0, 5.0]  # utility-level, corresponds to paper L in {0.2,...,1.0} with r=5
    all_rows: List[Dict] = []

    for n in [100, 1000]:
        ks = compute_k_values(n)
        all_rows.extend(
            run_linear_block(
                n=n,
                ks=ks,
                theorem_bounds=theorem_bounds,
            )
        )
        all_rows.extend(
            run_softmax_block(
                n=n,
                ks=ks,
                theorem_bounds=theorem_bounds,
                search_samples=args.softmax_search_samples,
                final_samples=args.softmax_final_samples,
                batch_size=args.softmax_batch_size,
                seed=args.seed,
            )
        )

    summary_csv = RESULTS / "global_smoothness_summary.csv"
    write_summary(all_rows, summary_csv)

    plot_global_smoothness_2x1(
        all_rows,
        n=100,
        ykey="best_empirical_L",
        ylabel="Empirical Smoothness",
        title="Empirical Smoothness vs Target L (n=100)",
        out_pdf=str(FIGURES / "global_smoothness_n100_empirical_vs_targetL.pdf"),
    )
    plot_global_smoothness_2x1(
        all_rows,
        n=1000,
        ykey="best_empirical_L",
        ylabel="Empirical Smoothness",
        title="Empirical Smoothness vs Target L (n=1000)",
        out_pdf=str(FIGURES / "global_smoothness_n1000_empirical_vs_targetL.pdf"),
    )
    plot_global_smoothness_2x1(
        all_rows,
        n=100,
        ykey="ratio_empirical_to_targetL",
        ylabel="Empirical / Target",
        title="Empirical / Target L (n=100)",
        out_pdf=str(FIGURES / "global_smoothness_n100_ratio_vs_targetL.pdf"),
    )
    plot_global_smoothness_2x1(
        all_rows,
        n=1000,
        ykey="ratio_empirical_to_targetL",
        ylabel="Empirical / Target",
        title="Empirical / Target L (n=1000)",
        out_pdf=str(FIGURES / "global_smoothness_n1000_ratio_vs_targetL.pdf"),
    )
    eps_csv, eps_pdf = _epsilon_response_panels(
        all_rows,
        seed=args.seed,
        softmax_samples=args.epsilon_softmax_samples,
        batch_size=args.softmax_batch_size,
    )

    print(f"Saved summary: {summary_csv}")
    print("Saved figures:")
    for name in [
        "global_smoothness_n100_empirical_vs_targetL.pdf",
        "global_smoothness_n1000_empirical_vs_targetL.pdf",
        "global_smoothness_n100_ratio_vs_targetL.pdf",
        "global_smoothness_n1000_ratio_vs_targetL.pdf",
        "epsilon_response_n100_k10_L1.pdf",
    ]:
        print(f"  {FIGURES / name}")
    print(f"Saved epsilon CSV: {eps_csv}")


if __name__ == "__main__":
    main()
