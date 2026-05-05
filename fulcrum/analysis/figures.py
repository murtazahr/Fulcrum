"""Figure generation for the manuscript.

Three plot types:

- :func:`plot_pareto_setting` — privacy–utility Pareto for one setting,
  topology-aware vs uniform allocation, faceted by $T_{\\max}$.
- :func:`plot_eta_sweep` — η vs attack lift, with the IID-null at η=0.
- :func:`plot_attack_channel_ablation` — bar chart per channel
  ($\\mathcal{A}_1, \\mathcal{A}_2^{\\text{topo}}, \\mathcal{A}_2^{\\text{org}},
  \\mathcal{A}_2^{\\text{full}}$) showing attack lift.

Output: PNG (for inspection) + PDF (for manuscript inclusion). Style is kept
journal-clean (Times-like font, gridlines, 95% CI bands).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fulcrum.analysis.pareto import extract_pareto_per_group


# ---------------------------------------------------------------------------
# Plot styling — applied lazily to keep matplotlib import deferred
# ---------------------------------------------------------------------------

def _set_style():
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.bbox": "tight",
        "savefig.dpi": 150,
    })


def _save(fig, output_path: str | Path) -> None:
    """Save as both PNG and PDF (PDF for the manuscript, PNG for quick viewing)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix(".png"))
    fig.savefig(output_path.with_suffix(".pdf"))


# ---------------------------------------------------------------------------
# Per-setting Pareto figure
# ---------------------------------------------------------------------------

def plot_pareto_setting(
    df: pd.DataFrame,
    setting: str,
    output_path: str | Path,
    x_col: str = "K_star",
    y_col: str = "final_loss",
) -> None:
    """Generate the Pareto-frontier figure for one setting.

    Faceted by ``dp.observation_window`` (one subplot per $T_{\\max}$). Within
    each subplot, two curves: topology-aware and uniform allocation.
    """
    _set_style()
    import matplotlib.pyplot as plt

    df = df.dropna(subset=[x_col, y_col, "dp.allocation", "dp.observation_window"])
    if df.empty:
        raise ValueError(f"No data for Pareto plot in setting {setting}")

    t_max_values = sorted(df["dp.observation_window"].unique())
    fig, axes = plt.subplots(1, len(t_max_values), figsize=(4 * len(t_max_values), 3.2),
                             sharey=True)
    if len(t_max_values) == 1:
        axes = [axes]

    color_map = {"topology_aware": "#1f77b4", "uniform": "#d62728"}
    marker_map = {"topology_aware": "o", "uniform": "s"}

    for ax, t_max in zip(axes, t_max_values):
        sub = df[df["dp.observation_window"] == t_max]
        fronts = extract_pareto_per_group(sub, group_col="dp.allocation", x_col=x_col, y_col=y_col)
        # Plot raw points (faint) + Pareto curve (bold)
        for alloc, sub2 in sub.groupby("dp.allocation"):
            color = color_map.get(alloc, "gray")
            ax.scatter(sub2[x_col], sub2[y_col], color=color, alpha=0.25, s=20)
            if alloc in fronts and len(fronts[alloc]) >= 2:
                front = fronts[alloc].sort_values(x_col)
                ax.plot(front[x_col], front[y_col], color=color, marker=marker_map.get(alloc, "o"),
                        label=alloc, linewidth=1.8, markersize=5)
        ax.set_title(f"$T_{{\\max}}$ = {int(t_max)}")
        ax.set_xlabel(r"Privacy bound $K^\star$ (nats)")
        ax.legend(loc="upper right")

    axes[0].set_ylabel("Test loss")
    fig.suptitle(f"Setting {setting}: Privacy-Utility Pareto Frontier", y=1.02)
    _save(fig, output_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# η-sweep figure (Setting C central experiment)
# ---------------------------------------------------------------------------

def plot_eta_sweep(
    df: pd.DataFrame,
    output_path: str | Path,
    metric_col: str = "K_star",
    eta_col: str = "data.eta",
    allocation_col: str = "dp.allocation",
) -> None:
    """Plot $K^\\star$ vs η for topology-aware vs uniform.

    Under Theorem 2 + Corollary 3, the gap between the two curves should grow
    monotonically with η. At η=0 (IID null) they should coincide.
    """
    _set_style()
    import matplotlib.pyplot as plt

    df = df.dropna(subset=[metric_col, eta_col, allocation_col])
    fig, ax = plt.subplots(figsize=(5, 3.5))

    color_map = {"topology_aware": "#1f77b4", "uniform": "#d62728"}
    for alloc, sub in df.groupby(allocation_col):
        agg = sub.groupby(eta_col)[metric_col].agg(["mean", "std", "count"]).reset_index()
        ci = 1.96 * agg["std"] / np.sqrt(agg["count"].clip(lower=1))
        ax.plot(agg[eta_col], agg["mean"], marker="o", label=alloc, color=color_map.get(alloc, "gray"))
        ax.fill_between(agg[eta_col], agg["mean"] - ci, agg["mean"] + ci, alpha=0.2,
                        color=color_map.get(alloc, "gray"))

    ax.set_xlabel(r"Topology-data coupling $\eta$")
    ax.set_ylabel(r"Worst-case privacy bound $K^\star$ (nats)")
    ax.set_title("Setting C: Allocation gap grows with leverage")
    ax.legend()
    _save(fig, output_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Channel-ablation bar chart
# ---------------------------------------------------------------------------

def plot_attack_channel_ablation(
    df: pd.DataFrame,
    output_path: str | Path,
    metric_col: str = "attack_lift",
    channel_col: str = "channel",
    setting_col: str = "setting",
) -> None:
    """Bar chart of attack lift per channel, grouped by setting.

    Expects ``df`` to be the *attack-evaluation* DataFrame (one row per
    (target_run, channel) pair), not the raw runs DataFrame.
    """
    _set_style()
    import matplotlib.pyplot as plt

    df = df.dropna(subset=[metric_col, channel_col, setting_col])
    settings = sorted(df[setting_col].unique())
    channels = ["A1", "A2_topo", "A2_org", "A2_full"]
    width = 0.2
    fig, ax = plt.subplots(figsize=(7, 3.5))

    for i, channel in enumerate(channels):
        means, errs = [], []
        for s in settings:
            sub = df[(df[setting_col] == s) & (df[channel_col] == channel)]
            if sub.empty:
                means.append(0.0)
                errs.append(0.0)
            else:
                means.append(float(sub[metric_col].mean()))
                errs.append(1.96 * float(sub[metric_col].std()) / np.sqrt(max(len(sub), 1)))
        x_pos = np.arange(len(settings)) + (i - 1.5) * width
        ax.bar(x_pos, means, width, yerr=errs, label=channel, capsize=2)

    ax.set_xticks(np.arange(len(settings)))
    ax.set_xticklabels([f"Setting {s}" for s in settings])
    ax.set_ylabel("Attack lift = baseline − loss")
    ax.set_title("TADI channel ablations")
    ax.legend(loc="upper right")
    ax.axhline(0, color="black", linewidth=0.5)
    _save(fig, output_path)
    plt.close(fig)
