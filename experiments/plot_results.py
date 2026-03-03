"""Plotting utilities for regret and smoothness experiments."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.lines import Line2D


# Consistent style
COLORS = {
    "Linear Lottery": "#1f77b4",
    "Softmax": "#ff7f0e",
    "MERIT": "#2ca02c",
    "Swiss NSF": "#d62728",
    "lower_bound": "#888888",
}
plt.rcParams.update({"font.size": 12, "figure.figsize": (7, 4.5)})

AXIS_LABEL_SIZE = 15
TICK_LABEL_SIZE = 12
LINE_WIDTH = 2.2
MARKER_SIZE = 6
CAP_SIZE = 3
LINE_STYLES = {"Linear Lottery": "-", "Softmax": "--"}
MARKERS = {"Linear Lottery": "o", "Softmax": "s"}


def _set_decimal_log_ticks(ax, x_values) -> None:
    """Use decimal labels on log-scaled L axes instead of 10^k notation."""
    x = np.asarray(x_values, dtype=float)
    if x.size == 0:
        return
    # For the standard experiment grids (<=~20 values), label each sampled L.
    if x.size <= 25:
        ax.xaxis.set_major_locator(ticker.FixedLocator(x))
        labels = [f"{v:.2f}" if v < 1 else f"{v:.2f}".rstrip("0").rstrip(".") for v in x]
        ax.xaxis.set_major_formatter(ticker.FixedFormatter(labels))
        ax.tick_params(axis="x", labelrotation=45, labelsize=9)
    else:
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))


# ---------------------------------------------------------------------------
# Regret plots
# ---------------------------------------------------------------------------

def plot_regret_vs_param(
    results: dict, param_name: str,
    title: str = "", save_path: str = None,
    logx: bool = False,
    include_bounds: bool = True,
    x_label: str = None,
) -> None:
    """Line plot: regret of linear lottery and softmax vs a parameter.

    results must contain keys: param_name, regret_linear, regret_softmax,
    bound_linear, bound_softmax, lower_bound.
    """
    fig, ax = plt.subplots()
    x = results[param_name]

    if "regret_linear_std" in results:
        ax.errorbar(x, results["regret_linear"], yerr=results["regret_linear_std"],
                    fmt=f"{MARKERS['Linear Lottery']}{LINE_STYLES['Linear Lottery']}",
                    color=COLORS["Linear Lottery"], label="Linear Lottery",
                    markersize=MARKER_SIZE, capsize=CAP_SIZE, elinewidth=1.2, linewidth=LINE_WIDTH)
    else:
        ax.plot(x, results["regret_linear"],
                f"{MARKERS['Linear Lottery']}{LINE_STYLES['Linear Lottery']}",
                color=COLORS["Linear Lottery"],
                label="Linear Lottery", markersize=MARKER_SIZE, linewidth=LINE_WIDTH)
    if "regret_softmax_std" in results:
        ax.errorbar(x, results["regret_softmax"], yerr=results["regret_softmax_std"],
                    fmt=f"{MARKERS['Softmax']}{LINE_STYLES['Softmax']}",
                    color=COLORS["Softmax"], label="Softmax",
                    markersize=MARKER_SIZE, capsize=CAP_SIZE, elinewidth=1.2, linewidth=LINE_WIDTH)
    else:
        ax.plot(x, results["regret_softmax"],
                f"{MARKERS['Softmax']}{LINE_STYLES['Softmax']}",
                color=COLORS["Softmax"],
                label="Softmax", markersize=MARKER_SIZE, linewidth=LINE_WIDTH)
    if include_bounds:
        ax.plot(x, results["bound_linear"], "--", color=COLORS["Linear Lottery"],
                alpha=0.5, label="Linear bound (Thm 2)")
        ax.fill_between(x, 0, results["lower_bound"], color=COLORS["lower_bound"],
                         alpha=0.15, label="Lower bound (Thm 3)")

    # error bars if available
    for key, color in [
        ("regret_linear_std", COLORS["Linear Lottery"]),
        ("regret_softmax_std", COLORS["Softmax"]),
    ]:
        if key in results:
            y = results[key.replace("_std", "")]
            ax.fill_between(x,
                            np.array(y) - np.array(results[key]),
                            np.array(y) + np.array(results[key]),
                            color=color, alpha=0.15)

    if logx:
        ax.set_xscale("log")
        if param_name == "L":
            _set_decimal_log_ticks(ax, x)
    ax.set_xlabel(x_label if x_label is not None else
                  ("Smoothness parameter $L$" if param_name == "L" else param_name),
                  fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Regret", fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    ax.set_title(title)
    ax.legend(fontsize=9)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_regret_vs_param_sidebyside(
    results_list: list,
    param_name: str,
    subtitles: list,
    suptitle: str = "",
    save_path: str = None,
    logx: bool = False,
    include_bounds: bool = True,
    x_label: str = None,
    y_label: str = "Regret",
    ncols: int = None,
    sharey: bool = True,
) -> None:
    """Side-by-side subplots of regret vs a parameter for different configs.

    Parameters
    ----------
    results_list : list of dicts (one per subplot)
    subtitles : list of str (one per subplot)
    """
    n_panels = len(results_list)
    if ncols is None:
        ncols = n_panels
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6 * ncols, 4.2 * nrows), sharey=sharey
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.ravel().tolist()

    for ax, results, subtitle in zip(axes, results_list, subtitles):
        x = results[param_name]
        if "regret_linear_std" in results:
            ax.errorbar(x, results["regret_linear"], yerr=results["regret_linear_std"],
                        fmt=f"{MARKERS['Linear Lottery']}{LINE_STYLES['Linear Lottery']}",
                        color=COLORS["Linear Lottery"], label="Linear Lottery",
                        markersize=MARKER_SIZE, capsize=CAP_SIZE, elinewidth=1.2, linewidth=LINE_WIDTH)
        else:
            ax.plot(x, results["regret_linear"],
                    f"{MARKERS['Linear Lottery']}{LINE_STYLES['Linear Lottery']}",
                    color=COLORS["Linear Lottery"],
                    label="Linear Lottery", markersize=MARKER_SIZE, linewidth=LINE_WIDTH)
        if "regret_softmax_std" in results:
            ax.errorbar(x, results["regret_softmax"], yerr=results["regret_softmax_std"],
                        fmt=f"{MARKERS['Softmax']}{LINE_STYLES['Softmax']}",
                        color=COLORS["Softmax"], label="Softmax",
                        markersize=MARKER_SIZE, capsize=CAP_SIZE, elinewidth=1.2, linewidth=LINE_WIDTH)
        else:
            ax.plot(x, results["regret_softmax"],
                    f"{MARKERS['Softmax']}{LINE_STYLES['Softmax']}",
                    color=COLORS["Softmax"],
                    label="Softmax", markersize=MARKER_SIZE, linewidth=LINE_WIDTH)
        if include_bounds:
            ax.plot(x, results["bound_linear"], "--", color=COLORS["Linear Lottery"],
                    alpha=0.5, label="Linear bound (Thm 2)")
            ax.fill_between(x, 0, results["lower_bound"], color=COLORS["lower_bound"],
                             alpha=0.15, label="Lower bound (Thm 3)")

        for key, color in [
            ("regret_linear_std", COLORS["Linear Lottery"]),
            ("regret_softmax_std", COLORS["Softmax"]),
        ]:
            if key in results:
                y = results[key.replace("_std", "")]
                ax.fill_between(x,
                                np.array(y) - np.array(results[key]),
                                np.array(y) + np.array(results[key]),
                                color=color, alpha=0.15)

        if logx:
            ax.set_xscale("log")
            if param_name == "L":
                _set_decimal_log_ticks(ax, x)
        ax.set_xlabel(x_label if x_label is not None else
                      ("Smoothness parameter $L$" if param_name == "L" else param_name),
                      fontsize=AXIS_LABEL_SIZE)
        ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
        ax.set_title(subtitle)

    # Hide any extra axes if grid has spare cells.
    for ax in axes[len(results_list):]:
        ax.set_visible(False)

    for ax in axes[:len(results_list)]:
        ax.set_ylim(bottom=0)

    # Set y-label for the left-most subplot in each row.
    for row in range(nrows):
        idx = row * ncols
        if idx < len(results_list):
            axes[idx].set_ylabel(y_label, fontsize=AXIS_LABEL_SIZE)
    legend_handles = [
        Line2D([0], [0], color=COLORS["Linear Lottery"], marker=MARKERS["Linear Lottery"],
               linestyle=LINE_STYLES["Linear Lottery"], label="Linear Lottery"),
        Line2D([0], [0], color=COLORS["Softmax"], marker=MARKERS["Softmax"],
               linestyle=LINE_STYLES["Softmax"], label="Softmax"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        fontsize=16,
        frameon=True,
        edgecolor="black",
        fancybox=False,
        framealpha=1.0,
    )
    if suptitle:
        fig.suptitle(suptitle, y=1.02, fontsize=12)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Smoothness plots
# ---------------------------------------------------------------------------

def plot_smoothness_bars(
    results: dict,
    L_theoretical: float = None,
    title: str = "",
    save_path: str = None,
) -> None:
    """Grouped bar chart: worst-case and mean smoothness per mechanism."""
    names = list(results.keys())
    worst = [results[n]["worst_case"] for n in names]
    mean = [results[n]["mean"] for n in names]

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots()

    ax.bar(x - width / 2, worst, width, label="Worst-case", color="#d62728", alpha=0.8)
    ax.bar(x + width / 2, mean, width, label="Mean", color="#1f77b4", alpha=0.8)

    if L_theoretical is not None:
        ax.axhline(L_theoretical, ls="--", color="black", alpha=0.6,
                    label=f"Theoretical L={L_theoretical}")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Lipschitz constant")
    ax.set_title(title)
    ax.legend(fontsize=9)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_smoothness_vs_L(
    results: dict,
    title: str = "",
    save_path: str = None,
) -> None:
    """Empirical smoothness vs theoretical L for linear lottery and softmax."""
    L = results["L"]
    fig, ax = plt.subplots()

    ax.plot(L, L, "k--", alpha=0.4, label="y = L (theoretical)")
    ax.plot(L, results["linear_worst"], "o-", color=COLORS["Linear Lottery"],
            markersize=4, label="Linear Lottery (worst)")
    ax.plot(L, results["linear_mean"], "o--", color=COLORS["Linear Lottery"],
            markersize=3, alpha=0.6, label="Linear Lottery (mean)")
    ax.plot(L, results["softmax_worst"], "s-", color=COLORS["Softmax"],
            markersize=4, label="Softmax (worst)")
    ax.plot(L, results["softmax_mean"], "s--", color=COLORS["Softmax"],
            markersize=3, alpha=0.6, label="Softmax (mean)")

    ax.set_xlabel("Target smoothness L")
    ax.set_ylabel("Empirical Lipschitz constant")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_smoothness_histogram(
    results: dict,
    title: str = "",
    save_path: str = None,
) -> None:
    """Overlaid histograms of local Lipschitz values per mechanism."""
    fig, ax = plt.subplots()

    for name, r in results.items():
        vals = r["values"]
        color = COLORS.get(name, None)
        ax.hist(vals, bins=30, alpha=0.4, label=name, color=color, density=True)

    ax.set_xlabel("Local Lipschitz constant")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(fontsize=9)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Beta sweep + utility CCDF + global smoothness plots
# ---------------------------------------------------------------------------

def plot_beta_sweep(
    df,
    out_pdf: str,
    k_name: str,
    colors: dict = None,
    linestyles: dict = None,
) -> None:
    colors = colors or {"Linear Lottery": "#1f77b4", "Softmax": "#ff7f0e"}
    linestyles = linestyles or {"Linear Lottery": "-", "Softmax": "--"}

    l_mults = sorted(df["L_multiplier"].unique())
    ncols = len(l_mults)
    fig, axes = plt.subplots(1, ncols, figsize=(6.0 * ncols, 4.4), sharey=False, constrained_layout=True)
    if ncols == 1:
        axes = [axes]

    legend_handles = None
    legend_labels = None
    for ax, l_mult in zip(axes, l_mults):
        sub = df[df["L_multiplier"] == l_mult]
        g = (
            sub.groupby("alpha", as_index=False)
            .agg(
                linear_mean=("regret_linear_per_k", "mean"),
                linear_std=("regret_linear_per_k", "std"),
                softmax_mean=("regret_softmax_per_k", "mean"),
                softmax_std=("regret_softmax_per_k", "std"),
            )
            .sort_values("alpha")
        )
        x = g["alpha"].to_numpy()
        ax.errorbar(
            x,
            g["linear_mean"].to_numpy(),
            yerr=np.nan_to_num(g["linear_std"].to_numpy()),
            color=colors["Linear Lottery"],
            linestyle=linestyles["Linear Lottery"],
            marker="o",
            linewidth=2.4,
            markersize=6.5,
            capsize=3.5,
            label="Linear Lottery",
        )
        ax.errorbar(
            x,
            g["softmax_mean"].to_numpy(),
            yerr=np.nan_to_num(g["softmax_std"].to_numpy()),
            color=colors["Softmax"],
            linestyle=linestyles["Softmax"],
            marker="s",
            linewidth=2.4,
            markersize=6.5,
            capsize=3.5,
            label="Softmax",
        )
        ax.set_xlabel(r"Symmetric Beta Shape $\alpha=\beta$", fontsize=12)
        ax.set_ylabel("Regret / k", fontsize=12)
        ax.set_ylim(bottom=0)
        ax.set_xticks(x)
        ax.set_title(rf"$L={l_mult:.1f}\cdot(1/r)$", fontsize=13)
        ax.grid(alpha=0.25, linestyle=":")
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=2,
            frameon=True,
            fontsize=10,
        )
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_utility_ccdf_curves(
    curves: list,
    out_pdf: str,
    x_label: str = "Utility (normalized to [0,1])",
    y_label: str = "CCDF: P(U > u)",
) -> None:
    """curves: list of dict with keys x, y, label, color."""
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    for c in curves:
        ax.plot(c["x"], c["y"], label=c["label"], color=c["color"], linewidth=2.4)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_global_smoothness_2x1(
    rows: list,
    n: int,
    ykey: str,
    ylabel: str,
    title: str,
    out_pdf: str,
) -> None:
    """rows: merged summary rows (dicts) from global smoothness run.

    Despite the historical name, this now renders a 1x2 layout.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), sharey=True, constrained_layout=True)
    mech_order = ["linear", "softmax"]
    mech_label = {"linear": "Linear Lottery", "softmax": "Softmax"}
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
    linestyles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "D", "P", "X", "v", "*"]

    subset = [r for r in rows if int(r["n"]) == n]
    legend_handles = None
    legend_labels = None
    for ax, mech in zip(axes, mech_order):
        mrows = [r for r in subset if r["mechanism"] == mech]
        ks = sorted({int(r["k"]) for r in mrows})
        for i, k in enumerate(ks):
            s = sorted([r for r in mrows if int(r["k"]) == k], key=lambda z: float(z["L"]))
            x = [float(r["L"]) for r in s]
            y = [float(r[ykey]) for r in s]
            ax.plot(
                x,
                y,
                color=colors[i % len(colors)],
                linestyle=linestyles[i % len(linestyles)],
                marker=markers[i % len(markers)],
                linewidth=2.6,
                markersize=7.5,
                markeredgewidth=1.0,
                label=f"k={k}",
            )
        if ykey == "ratio_empirical_to_targetL":
            ax.axhline(1.0, linestyle="--", color="black", linewidth=1.2, alpha=0.8, label="_nolegend_")
        else:
            maxv = max([max(float(r["L"]), float(r[ykey])) for r in mrows], default=1.0)
            ax.plot([0, maxv * 1.05], [0, maxv * 1.05], linestyle="--", color="black", linewidth=1.2, alpha=0.8, label="_nolegend_")
        ax.set_title(mech_label[mech])
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25, linestyle=":")
        ax.set_ylim(bottom=0)
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
    for ax in axes:
        ax.set_xlabel("Target L")
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.03),
            ncol=max(1, min(3, len(legend_labels))),
            frameon=True,
        )
    # No figure title for paper-ready exports.
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_local_sensitivity_2x1(summary_df, k_name: str, out_pdf: str) -> None:
    """1x2 local-sensitivity plot (Linear left, Softmax right) across datasets.

    summary_df columns required:
    - dataset, mechanism, L, local_sensitivity
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), sharey=True, constrained_layout=True)
    mech_order = ["linear", "softmax"]
    mech_label = {"linear": "Linear Lottery", "softmax": "Softmax"}
    ds_colors = {
        "Beta": "#2ca02c",
        "ICLR": "#1f77b4",
        "NeurIPS": "#ff7f0e",
        "Swiss NSF": "#d62728",
    }
    markers = ["o", "s", "^", "D", "P", "X"]
    linestyles = ["-", "--", "-.", ":"]

    datasets = sorted(summary_df["dataset"].unique())
    legend_handles = None
    legend_labels = None
    for ax, mech in zip(axes, mech_order):
        sub = summary_df[summary_df["mechanism"] == mech]
        for i, ds in enumerate(datasets):
            ds_sub = sub[sub["dataset"] == ds].sort_values("L")
            if ds_sub.empty:
                continue
            ax.plot(
                ds_sub["L"].to_numpy(),
                ds_sub["local_sensitivity"].to_numpy(),
                color=ds_colors.get(ds, COLORS["Linear Lottery"]),
                marker=markers[i % len(markers)],
                linestyle=linestyles[i % len(linestyles)],
                linewidth=2.4,
                markersize=6.8,
                label=ds,
            )
        ax.plot([0.2, 1.0], [0.2, 1.0], linestyle="--", color="black", linewidth=1.2, alpha=0.8, label="_nolegend_")
        ax.set_title(mech_label[mech])
        ax.set_ylabel("Local Sensitivity")
        ax.grid(alpha=0.25, linestyle=":")
        ax.set_ylim(bottom=0)
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
    for ax in axes:
        ax.set_xlabel("Target L")
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.03),
            ncol=max(1, min(4, len(legend_labels))),
            frameon=True,
        )
    # No figure title for paper-ready exports.
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_local_smoothness_2x1(summary_df, k_name: str, out_pdf: str) -> None:
    """Backward-compatible wrapper; use `plot_local_sensitivity_2x1`."""
    return plot_local_sensitivity_2x1(summary_df=summary_df, k_name=k_name, out_pdf=out_pdf)


def plot_baseline_local_sensitivity_panels(summary_df, out_pdf: str) -> None:
    """1xN panels over k_name with dataset x-axis and mechanism bars."""
    k_names = list(summary_df["k_name"].drop_duplicates())
    ncols = len(k_names)
    fig, axes = plt.subplots(1, ncols, figsize=(5.2 * ncols, 4.4), sharey=True, constrained_layout=True)
    if ncols == 1:
        axes = [axes]

    mech_order = ["MERIT", "Swiss NSF", "Randomized Threshold"]
    mech_colors = {
        "MERIT": "#2ca02c",
        "Swiss NSF": "#d62728",
        "Randomized Threshold": "#1f77b4",
    }
    datasets = sorted(summary_df["dataset"].unique())
    x = np.arange(len(datasets))
    width = 0.23

    legend_handles = None
    legend_labels = None
    for ax, k_name in zip(axes, k_names):
        sub = summary_df[summary_df["k_name"] == k_name]
        for j, mech in enumerate(mech_order):
            vals = []
            for ds in datasets:
                t = sub[(sub["dataset"] == ds) & (sub["mechanism"] == mech)]
                vals.append(float(t["local_sensitivity"].iloc[0]) if len(t) else np.nan)
            bars = ax.bar(
                x + (j - 1) * width,
                vals,
                width=width,
                color=mech_colors[mech],
                alpha=0.9,
                label=mech,
            )
            if legend_handles is None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()
        ax.set_title(k_name)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=20)
        ax.set_xlabel("Dataset")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.25, linestyle=":")

    axes[0].set_ylabel("Worst-Case Local Sensitivity")
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=3,
            frameon=True,
        )
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
