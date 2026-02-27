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
                    fmt="o-", color=COLORS["Linear Lottery"], label="Linear Lottery",
                    markersize=MARKER_SIZE, capsize=CAP_SIZE, elinewidth=1.2, linewidth=LINE_WIDTH)
    else:
        ax.plot(x, results["regret_linear"], "o-", color=COLORS["Linear Lottery"],
                label="Linear Lottery", markersize=MARKER_SIZE, linewidth=LINE_WIDTH)
    if "regret_softmax_std" in results:
        ax.errorbar(x, results["regret_softmax"], yerr=results["regret_softmax_std"],
                    fmt="s--", color=COLORS["Softmax"], label="Softmax",
                    markersize=MARKER_SIZE, capsize=CAP_SIZE, elinewidth=1.2, linewidth=LINE_WIDTH)
    else:
        ax.plot(x, results["regret_softmax"], "s--", color=COLORS["Softmax"],
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
                        fmt="o-", color=COLORS["Linear Lottery"], label="Linear Lottery",
                        markersize=MARKER_SIZE, capsize=CAP_SIZE, elinewidth=1.2, linewidth=LINE_WIDTH)
        else:
            ax.plot(x, results["regret_linear"], "o-", color=COLORS["Linear Lottery"],
                    label="Linear Lottery", markersize=MARKER_SIZE, linewidth=LINE_WIDTH)
        if "regret_softmax_std" in results:
            ax.errorbar(x, results["regret_softmax"], yerr=results["regret_softmax_std"],
                        fmt="s--", color=COLORS["Softmax"], label="Softmax",
                        markersize=MARKER_SIZE, capsize=CAP_SIZE, elinewidth=1.2, linewidth=LINE_WIDTH)
        else:
            ax.plot(x, results["regret_softmax"], "s--", color=COLORS["Softmax"],
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
        Line2D([0], [0], color=COLORS["Linear Lottery"], marker="o", linestyle="-", label="Linear Lottery"),
        Line2D([0], [0], color=COLORS["Softmax"], marker="s", linestyle="--", label="Softmax"),
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
