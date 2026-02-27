"""Experiment 2: Empirical smoothness estimation.

Estimate expected and worst-case smoothness of all mechanisms by
perturbing review matrix entries and measuring ||p(X') - p(X)||_1 / epsilon.
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

_BASE = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, _BASE)

from mechanisms import (
    linear_lottery_mechanism,
    softmax_mechanism,
    merit_mechanism,
    swiss_nsf_mechanism,
    randomize_above_threshold,
)
from data_utils import (
    load_review_matrix,
    generate_gaussian_reviews,
    generate_constant_reviews,
)
from plot_results import plot_smoothness_bars, plot_smoothness_vs_L, plot_smoothness_histogram


# ---------------------------------------------------------------------------
# Core smoothness estimation
# ---------------------------------------------------------------------------

def estimate_local_smoothness(
    mechanism_fn: callable,
    X: np.ndarray, k: int,
    epsilon: float = 1e-3,
    n_perturbations: int = None,
    rng: np.random.Generator = None,
    **mechanism_kwargs,
) -> np.ndarray:
    """Estimate local Lipschitz constants by perturbing entries of X.

    For each perturbation (i, j): set X'_{ij} = X_{ij} + epsilon,
    compute p' = mechanism(X'), Lipschitz = ||p' - p||_1 / epsilon.

    Parameters
    ----------
    mechanism_fn : callable(X, k, **kwargs) -> ndarray
    X : review matrix (n, r)
    k : budget
    epsilon : perturbation size
    n_perturbations : number of (i,j) entries to perturb.
        If None, perturb all non-NaN entries (exhaustive).
    rng : random generator for selecting perturbation entries

    Returns
    -------
    lipschitz_values : ndarray of local Lipschitz estimates
    """
    if rng is None:
        rng = np.random.default_rng()

    p_base = mechanism_fn(X, k, **mechanism_kwargs)

    # identify non-NaN entries
    valid = np.argwhere(~np.isnan(X))
    if n_perturbations is None or n_perturbations >= len(valid):
        entries = valid
    else:
        idx = rng.choice(len(valid), size=n_perturbations, replace=False)
        entries = valid[idx]

    lipschitz_values = []
    for i, j in entries:
        X_pert = X.copy()
        X_pert[i, j] += epsilon
        p_pert = mechanism_fn(X_pert, k, **mechanism_kwargs)
        lip = float(np.abs(p_pert - p_base).sum()) / epsilon
        lipschitz_values.append(lip)

    return np.array(lipschitz_values)


def estimate_boundary_smoothness(
    mechanism_fn: callable,
    X: np.ndarray, k: int,
    margin: float = 1e-4,
    **mechanism_kwargs,
) -> np.ndarray:
    """Estimate worst-case smoothness by perturbing papers just past rank boundaries.

    For threshold-based methods, random perturbations rarely cross rank boundaries.
    Here we find each pair of adjacent-ranked papers, compute the minimal review
    perturbation that swaps their ranks, and measure the resulting probability change.

    The Lipschitz constant at the boundary is ||delta_p||_1 / delta, where delta
    is the review perturbation size (just enough to cross the boundary + margin).

    Returns array of Lipschitz values.
    """
    v = np.nanmean(X, axis=1)
    n, r = X.shape
    ranking = np.argsort(-v)
    p_base = mechanism_fn(X, k, **mechanism_kwargs)

    lipschitz_values = []

    for rank_pos in range(n - 1):
        i_above = ranking[rank_pos]
        i_below = ranking[rank_pos + 1]
        gap = v[i_above] - v[i_below]
        if gap <= 0:
            continue

        # To swap ranks, we need to change one review of i_below enough
        # so that its mean exceeds i_above's mean.
        # Changing X[i_below, j] by delta changes mean by delta / r_eff
        r_eff = np.sum(~np.isnan(X[i_below]))
        if r_eff == 0:
            continue
        delta = (gap + margin) * r_eff  # perturbation to one review entry

        for j in range(int(r_eff)):
            if np.isnan(X[i_below, j]):
                continue
            X_pert = X.copy()
            X_pert[i_below, j] += delta
            p_pert = mechanism_fn(X_pert, k, **mechanism_kwargs)
            change = float(np.abs(p_pert - p_base).sum())
            if change > 0:
                lipschitz_values.append(change / delta)
            break  # one review per boundary pair is enough

        # also try pushing i_above down
        r_eff_above = np.sum(~np.isnan(X[i_above]))
        if r_eff_above == 0:
            continue
        delta_above = (gap + margin) * r_eff_above
        for j in range(int(r_eff_above)):
            if np.isnan(X[i_above, j]):
                continue
            X_pert = X.copy()
            X_pert[i_above, j] -= delta_above
            p_pert = mechanism_fn(X_pert, k, **mechanism_kwargs)
            change = float(np.abs(p_pert - p_base).sum())
            if change > 0:
                lipschitz_values.append(change / delta_above)
            break

    return np.array(lipschitz_values) if lipschitz_values else np.array([0.0])


def estimate_smoothness_paired_softmax(
    X: np.ndarray, k: int, L: float,
    epsilon: float = 1e-3,
    n_perturbations: int = None,
    n_samples: int = 100_000,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Smoothness estimation for softmax with paired RNG to cancel MC noise.

    Uses the same Gumbel noise draws for X and X' so that the only
    difference in p comes from the utility perturbation.
    """
    if rng is None:
        rng = np.random.default_rng()

    # base probabilities with a fixed seed
    seed_base = rng.integers(2**31)
    p_base = softmax_mechanism(X, k, L, n_samples=n_samples,
                               rng=np.random.default_rng(seed_base))

    valid = np.argwhere(~np.isnan(X))
    if n_perturbations is None or n_perturbations >= len(valid):
        entries = valid
    else:
        idx = rng.choice(len(valid), size=n_perturbations, replace=False)
        entries = valid[idx]

    lipschitz_values = []
    for i, j in entries:
        X_pert = X.copy()
        X_pert[i, j] += epsilon
        p_pert = softmax_mechanism(X_pert, k, L, n_samples=n_samples,
                                    rng=np.random.default_rng(seed_base))
        lip = float(np.abs(p_pert - p_base).sum()) / epsilon
        lipschitz_values.append(lip)

    return np.array(lipschitz_values)


# ---------------------------------------------------------------------------
# Comparison runner
# ---------------------------------------------------------------------------

def run_smoothness_comparison(
    X: np.ndarray, k: int, L: float,
    epsilon: float = 1e-3,
    n_perturbations: int = 200,
    skip_merit: bool = True,
    n_merit_perturbations: int = 30,
    interval_method: str = "leave_one_out",
    rng: np.random.Generator = None,
) -> dict:
    """Compare smoothness of all mechanisms on a single dataset.

    Returns dict: {mechanism_name: {"worst_case": float, "mean": float, "values": ndarray}}
    """
    if rng is None:
        rng = np.random.default_rng()

    results = {}

    # Linear lottery
    print("  Linear lottery...", end=" ", flush=True)
    t0 = time.time()
    vals = estimate_local_smoothness(
        linear_lottery_mechanism, X, k, epsilon=epsilon,
        n_perturbations=n_perturbations, rng=rng, L=L,
    )
    results["Linear Lottery"] = {
        "worst_case": float(vals.max()), "mean": float(vals.mean()), "values": vals,
    }
    print(f"({time.time()-t0:.1f}s)")

    # Softmax (paired)
    print("  Softmax...", end=" ", flush=True)
    t0 = time.time()
    vals = estimate_smoothness_paired_softmax(
        X, k, L, epsilon=epsilon, n_perturbations=n_perturbations,
        n_samples=100_000, rng=rng,
    )
    results["Softmax"] = {
        "worst_case": float(vals.max()), "mean": float(vals.mean()), "values": vals,
    }
    print(f"({time.time()-t0:.1f}s)")

    # Swiss NSF — use both random and boundary-targeted perturbations
    print("  Swiss NSF...", end=" ", flush=True)
    t0 = time.time()
    vals_rand = estimate_local_smoothness(
        swiss_nsf_mechanism, X, k, epsilon=epsilon,
        n_perturbations=n_perturbations, rng=rng,
        interval_method=interval_method,
    )
    vals_boundary = estimate_boundary_smoothness(
        swiss_nsf_mechanism, X, k, interval_method=interval_method,
    )
    vals = np.concatenate([vals_rand, vals_boundary])
    results["Swiss NSF"] = {
        "worst_case": float(vals.max()), "mean": float(vals_rand.mean()), "values": vals,
    }
    print(f"({time.time()-t0:.1f}s)")

    # Randomize-above-threshold (match linear lottery pool size)
    # estimate m from linear lottery: count candidates with 0 < p < 1
    p_lin = linear_lottery_mechanism(X, k, L)
    m_est = max(1, int(np.sum((p_lin > 1e-6) & (p_lin < 1 - 1e-6)) // 2))
    print(f"  Randomize-above-threshold (m={m_est})...", end=" ", flush=True)
    t0 = time.time()
    vals_rand = estimate_local_smoothness(
        randomize_above_threshold, X, k, epsilon=epsilon,
        n_perturbations=n_perturbations, rng=rng, m=m_est,
    )
    vals_boundary = estimate_boundary_smoothness(
        randomize_above_threshold, X, k, m=m_est,
    )
    vals = np.concatenate([vals_rand, vals_boundary])
    results[f"Rand-above-thresh (m={m_est})"] = {
        "worst_case": float(vals.max()), "mean": float(vals_rand.mean()), "values": vals,
    }
    print(f"({time.time()-t0:.1f}s)")

    # MERIT (expensive)
    if not skip_merit:
        print(f"  MERIT ({n_merit_perturbations} perturbations)...", end=" ", flush=True)
        t0 = time.time()
        vals = estimate_local_smoothness(
            merit_mechanism, X, k, epsilon=epsilon,
            n_perturbations=n_merit_perturbations, rng=rng,
            interval_method=interval_method,
        )
        results["MERIT"] = {
            "worst_case": float(vals.max()), "mean": float(vals.mean()), "values": vals,
        }
        print(f"({time.time()-t0:.1f}s)")

    return results


def run_smoothness_vs_L(
    X: np.ndarray, k: int, L_values: np.ndarray,
    epsilon: float = 1e-3,
    n_perturbations: int = 200,
    rng: np.random.Generator = None,
) -> dict:
    """Measure empirical smoothness of linear lottery and softmax across L values.

    Compare to theoretical guarantee (smoothness should be <= L).
    """
    if rng is None:
        rng = np.random.default_rng()

    results = {"L": [], "linear_worst": [], "linear_mean": [],
               "softmax_worst": [], "softmax_mean": []}

    for L in L_values:
        print(f"  L={L:.3f}...", end=" ", flush=True)

        vals_lin = estimate_local_smoothness(
            linear_lottery_mechanism, X, k, epsilon=epsilon,
            n_perturbations=n_perturbations, rng=rng, L=L,
        )
        vals_soft = estimate_smoothness_paired_softmax(
            X, k, L, epsilon=epsilon, n_perturbations=n_perturbations,
            n_samples=100_000, rng=rng,
        )

        results["L"].append(L)
        results["linear_worst"].append(float(vals_lin.max()))
        results["linear_mean"].append(float(vals_lin.mean()))
        results["softmax_worst"].append(float(vals_soft.max()))
        results["softmax_mean"].append(float(vals_soft.mean()))
        print(f"lin_max={vals_lin.max():.3f}, soft_max={vals_soft.max():.3f}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoothness experiments")
    parser.add_argument("--experiment", choices=["comparison", "vs_L"],
                        default="comparison")
    parser.add_argument("--data", choices=["neurips", "iclr", "gaussian", "constant"],
                        default="gaussian")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--r", type=int, default=5)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--L", type=float, default=2.0)
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument("--n_perturbations", type=int, default=200)
    parser.add_argument("--skip_merit", action="store_true")
    parser.add_argument("--n_subsample", type=int, default=150,
                        help="Subsample papers for MERIT feasibility")
    parser.add_argument("--output_dir", default=os.path.join(os.path.dirname(__file__), "results"))
    parser.add_argument("--fig_dir", default=os.path.join(os.path.dirname(__file__), "figures"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"seed={args.seed}, experiment={args.experiment}, data={args.data}")

    # Load / generate data
    if args.data == "neurips":
        X, _, _ = load_review_matrix("neurips2024", drop_rejected=True)
        label = "NeurIPS 2024"
    elif args.data == "iclr":
        X, _, _ = load_review_matrix("iclr2025", drop_rejected=True)
        label = "ICLR 2025"
    elif args.data == "gaussian":
        X, _ = generate_gaussian_reviews(n=args.n, r=args.r, rng=rng)
        label = f"Gaussian (n={args.n}, r={args.r})"
    elif args.data == "constant":
        X = generate_constant_reviews(n=args.n, r=args.r)
        label = f"Constant (n={args.n}, r={args.r})"

    n = X.shape[0]
    k = min(args.k, n // 2)

    # Subsample for large datasets (especially for MERIT)
    if not args.skip_merit and n > args.n_subsample:
        idx = rng.choice(n, size=args.n_subsample, replace=False)
        X_sub = X[idx]
        k_sub = max(1, int(k * args.n_subsample / n))
        print(f"Subsampled {n} -> {args.n_subsample} papers (k: {k} -> {k_sub})")
    else:
        X_sub = X
        k_sub = k

    if args.experiment == "comparison":
        print(f"\nSmootness comparison on {label} (n={X_sub.shape[0]}, k={k_sub}, L={args.L})")
        results = run_smoothness_comparison(
            X_sub, k_sub, args.L,
            epsilon=args.epsilon,
            n_perturbations=args.n_perturbations,
            skip_merit=args.skip_merit,
            rng=rng,
        )

        # Print summary table
        print(f"\n{'Mechanism':<30} {'Worst-case':>12} {'Mean':>12}")
        print("-" * 56)
        for name, r in results.items():
            print(f"{name:<30} {r['worst_case']:12.4f} {r['mean']:12.4f}")
        print(f"{'Theoretical L':30} {args.L:12.4f}")

        # Save
        rows = [{"mechanism": name, "worst_case": r["worst_case"], "mean": r["mean"]}
                for name, r in results.items()]
        df = pd.DataFrame(rows)
        out = os.path.join(args.output_dir, f"smoothness_comparison_{args.data}.csv")
        df.to_csv(out, index=False)
        print(f"\nSaved {out}")

        fig_path = os.path.join(args.fig_dir, f"smoothness_comparison_{args.data}.pdf")
        plot_smoothness_bars(results, L_theoretical=args.L,
                             title=f"Smoothness — {label} (k={k_sub})",
                             save_path=fig_path)
        print(f"Saved {fig_path}")

        fig_path2 = os.path.join(args.fig_dir, f"smoothness_hist_{args.data}.pdf")
        plot_smoothness_histogram(results, title=f"Local Lipschitz distribution — {label}",
                                  save_path=fig_path2)
        print(f"Saved {fig_path2}")

    elif args.experiment == "vs_L":
        L_values = np.logspace(-1, 0, 20)
        print(f"\nSmoothness vs L on {label} (n={X_sub.shape[0]}, k={k_sub})")
        results = run_smoothness_vs_L(
            X_sub, k_sub, L_values,
            epsilon=args.epsilon,
            n_perturbations=args.n_perturbations,
            rng=rng,
        )

        df = pd.DataFrame(results)
        out = os.path.join(args.output_dir, f"smoothness_vs_L_{args.data}.csv")
        df.to_csv(out, index=False)
        print(f"\nSaved {out}")

        fig_path = os.path.join(args.fig_dir, f"smoothness_vs_L_{args.data}.pdf")
        plot_smoothness_vs_L(results, title=f"Smoothness vs L — {label}",
                             save_path=fig_path)
        print(f"Saved {fig_path}")
