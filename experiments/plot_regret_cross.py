"""Replot cross-dataset regret figures from saved CSV outputs.

This script does not rerun mechanisms. It only reads CSVs in results/ and
regenerates figures in figures/ with current plotting style.
"""

import argparse
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd

from regret import load_dataset, compact_label, r_med
from plot_results import plot_regret_vs_param_sidebyside


def main(args):
    datasets = ["beta", "iclr", "neurips", "swissnsf"]
    k_names = ["k1", "k10pct", "k50pct"]

    os.makedirs(args.fig_dir, exist_ok=True)

    meta_args = SimpleNamespace(n=args.n, r=args.r)
    rng = np.random.default_rng(args.seed)
    meta = {}
    for ds in datasets:
        X, _, label, _, _, _ = load_dataset(ds, meta_args, rng)
        meta[ds] = (compact_label(label), X.shape[0], r_med(X))

    for k_name in k_names:
        results_list = []
        subtitles = []
        for ds in datasets:
            path = os.path.join(args.output_dir, f"regret_vs_L_cross_{ds}_{k_name}.csv")
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
            results_list.append(res)

        out_fig = os.path.join(args.fig_dir, f"regret_vs_L_cross_{k_name}.pdf")
        plot_regret_vs_param_sidebyside(
            results_list,
            "L",
            subtitles=subtitles,
            suptitle="",
            save_path=out_fig,
            logx=False,
            include_bounds=False,
            x_label=r"$L$",
            y_label="Regret / k" if args.normalize_by_k else "Regret",
            ncols=2,
            sharey=False,
        )
        print(f"Saved {out_fig}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replot cross-dataset regret figures from CSVs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=200, help="Beta synthetic n (for subtitle metadata)")
    parser.add_argument("--r", type=int, default=5, help="Beta synthetic reviews per item (for subtitle metadata)")
    parser.add_argument("--normalize_by_k", action="store_true", default=True)
    parser.add_argument("--no_normalize_by_k", action="store_false", dest="normalize_by_k")
    parser.add_argument("--output_dir", default=os.path.join(os.path.dirname(__file__), "results"))
    parser.add_argument("--fig_dir", default=os.path.join(os.path.dirname(__file__), "figures"))
    main(parser.parse_args())

