"""Tightness diagnostics for regret bounds.

Produces:
- ratio plots: regret / theoretical upper bound
- gap plots: (upper bound - regret)
for each k regime across the 4 datasets.
"""

import argparse
import math
import os

import numpy as np
import pandas as pd

from regret import (
    load_dataset,
    compact_label,
    r_min,
    r_med,
    k_configs,
    run_regret_vs_L,
    aggregate_trials,
)
from plot_results import plot_regret_vs_param_sidebyside


def linear_bound(k: int, n: int, d_v: float, L: np.ndarray) -> np.ndarray:
    return k * (1.0 - k / n) * d_v / (2.0 * L)


def softmax_bound(k: int, n: int, d_v: float, L: np.ndarray) -> np.ndarray:
    return (2.0 * d_v / math.e) * k * math.log(n) / L


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)

    base_rng = np.random.default_rng(args.seed)
    datasets = ["beta", "iclr", "neurips", "swissnsf"]

    loaded = {}
    for name in datasets:
        loaded[name] = load_dataset(name, args, base_rng)

    for k_name, k_fn in k_configs():
        ratio_panels = []
        ratio_subtitles = []
        gap_panels = []
        gap_subtitles = []

        for ds in datasets:
            X0, theta0, label, _, _, _ = loaded[ds]
            n = X0.shape[0]
            k = k_fn(n)
            rmin = r_min(X0)
            rmed = r_med(X0)
            d_v = 1.0 / rmin
            L_values = np.linspace(0.5 * d_v, 1.0, args.n_L_points)
            n_trials = args.beta_trials if ds == "beta" else 1

            trials = []
            for _ in range(n_trials):
                if ds == "beta":
                    trial_rng = np.random.default_rng(base_rng.integers(2**31))
                    X, theta, _, _, _, _ = load_dataset("beta", args, trial_rng)
                    run_rng = np.random.default_rng(base_rng.integers(2**31))
                else:
                    X, theta = X0, theta0
                    run_rng = base_rng
                trials.append(
                    run_regret_vs_L(
                        X, k, L_values, theta=theta,
                        n_softmax_samples=args.n_softmax_samples, rng=run_rng,
                    )
                )

            res = aggregate_trials(trials)
            L = np.asarray(res["L"], dtype=float)
            reg_lin = np.asarray(res["regret_linear"], dtype=float)
            reg_soft = np.asarray(res["regret_softmax"], dtype=float)
            b_lin = linear_bound(k, n, d_v, L)
            b_soft = softmax_bound(k, n, d_v, L)

            ratio_lin = np.divide(reg_lin, b_lin, out=np.full_like(reg_lin, np.nan), where=b_lin > 0)
            ratio_soft = np.divide(reg_soft, b_soft, out=np.full_like(reg_soft, np.nan), where=b_soft > 0)
            gap_lin = b_lin - reg_lin
            gap_soft = b_soft - reg_soft

            tight_df = pd.DataFrame({
                "L": L,
                "k": k,
                "n": n,
                "r_min": rmin,
                "r_med": rmed,
                "regret_linear": reg_lin,
                "regret_softmax": reg_soft,
                "bound_linear": b_lin,
                "bound_softmax": b_soft,
                "ratio_linear": ratio_lin,
                "ratio_softmax": ratio_soft,
                "gap_linear": gap_lin,
                "gap_softmax": gap_soft,
            })
            out_csv = os.path.join(args.output_dir, f"regret_tightness_{ds}_{k_name}.csv")
            tight_df.to_csv(out_csv, index=False)

            ratio_panel = {
                "L": L.tolist(),
                "regret_linear": ratio_lin.tolist(),
                "regret_softmax": ratio_soft.tolist(),
            }
            gap_panel = {
                "L": L.tolist(),
                "regret_linear": gap_lin.tolist(),
                "regret_softmax": gap_soft.tolist(),
            }
            if "regret_linear_std" in res:
                sd_lin = np.asarray(res["regret_linear_std"], dtype=float)
                sd_soft = np.asarray(res["regret_softmax_std"], dtype=float)
                ratio_panel["regret_linear_std"] = np.divide(sd_lin, b_lin, out=np.zeros_like(sd_lin), where=b_lin > 0).tolist()
                ratio_panel["regret_softmax_std"] = np.divide(sd_soft, b_soft, out=np.zeros_like(sd_soft), where=b_soft > 0).tolist()
                gap_panel["regret_linear_std"] = sd_lin.tolist()
                gap_panel["regret_softmax_std"] = sd_soft.tolist()

            subtitle = f"{compact_label(label)} (n={n}, r={rmed})"
            ratio_panels.append(ratio_panel)
            gap_panels.append(gap_panel)
            ratio_subtitles.append(subtitle)
            gap_subtitles.append(subtitle)

        ratio_fig = os.path.join(args.fig_dir, f"regret_tightness_ratio_{k_name}.pdf")
        plot_regret_vs_param_sidebyside(
            ratio_panels,
            "L",
            subtitles=ratio_subtitles,
            suptitle="",
            save_path=ratio_fig,
            logx=False,
            include_bounds=False,
            x_label=r"$L$",
            y_label="Regret / Upper bound",
            ncols=2,
            sharey=False,
        )

        gap_fig = os.path.join(args.fig_dir, f"regret_tightness_gap_{k_name}.pdf")
        plot_regret_vs_param_sidebyside(
            gap_panels,
            "L",
            subtitles=gap_subtitles,
            suptitle="",
            save_path=gap_fig,
            logx=False,
            include_bounds=False,
            x_label=r"$L$",
            y_label="Upper bound - Regret",
            ncols=2,
            sharey=False,
        )
        print(f"Saved {ratio_fig}")
        print(f"Saved {gap_fig}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regret tightness diagnostics")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=200, help="Beta synthetic n")
    parser.add_argument("--r", type=int, default=5, help="Beta synthetic reviews per item")
    parser.add_argument("--n_softmax_samples", type=int, default=10000)
    parser.add_argument("--beta_trials", type=int, default=20)
    parser.add_argument("--n_L_points", type=int, default=15)
    parser.add_argument("--output_dir", default=os.path.join(os.path.dirname(__file__), "results"))
    parser.add_argument("--fig_dir", default=os.path.join(os.path.dirname(__file__), "figures"))
    main(parser.parse_args())

