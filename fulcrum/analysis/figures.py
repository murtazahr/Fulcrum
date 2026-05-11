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

# Note: the strict Pareto-frontier extractor (`extract_pareto_per_group`) is
# still used by `topology_aware_advantage` in `pareto.py` for the
# area-between-curves headline number, but the figure itself plots per-$U$
# means rather than the dominance frontier (see `plot_pareto_setting`).


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
    budget_col: str = "dp.utility_budget_U",
) -> None:
    """Generate the privacy-utility figure for one setting.

    Faceted by ``dp.observation_window`` (one subplot per $T_{\\max}$). Within
    each subplot, two curves (topology-aware and uniform), each parameterised
    by the utility budget $U$. Curves are connected through the per-$U$ means;
    raw seed points are overlaid as faint scatter.

    Two scaling fixes vs. an earlier draft:

    1. ``sharey=False`` per panel — test loss is dominated by $T_{\\max}$
       (DP noise accumulates per round), so loss bands differ substantially
       across panels. Sharing the y-axis collapses each panel's in-band
       variation to a flat line.
    2. Connect per-$U$ means rather than the strict Pareto frontier: at fixed
       $T_{\\max}$, loss is near-flat in $K^\\star$, so the dominance
       algorithm extracts only 1-2 non-dominated points and the bold line is
       invisible. The actual signal is the horizontal shift between the TA
       and uniform curves at matched $U$, which mean-aggregation surfaces
       directly.
    """
    _set_style()
    import matplotlib.pyplot as plt

    needed = [x_col, y_col, "dp.allocation", "dp.observation_window", budget_col]
    df = df.dropna(subset=needed)
    if df.empty:
        raise ValueError(f"No data for Pareto plot in setting {setting}")

    t_max_values = sorted(df["dp.observation_window"].unique())
    fig, axes = plt.subplots(
        1, len(t_max_values),
        figsize=(4.2 * len(t_max_values), 3.6),
        sharey=False,  # let each panel auto-scale its loss range
    )
    if len(t_max_values) == 1:
        axes = [axes]

    color_map = {"topology_aware": "#1f77b4", "uniform": "#d62728"}
    marker_map = {"topology_aware": "o", "uniform": "s"}

    for ax, t_max in zip(axes, t_max_values):
        sub = df[df["dp.observation_window"] == t_max]
        for alloc, sub2 in sub.groupby("dp.allocation"):
            color = color_map.get(alloc, "gray")
            marker = marker_map.get(alloc, "o")
            # Raw seed points, faint
            ax.scatter(sub2[x_col], sub2[y_col], color=color, alpha=0.30, s=18,
                       edgecolors="none")
            # Per-U means with seed std as a vertical error bar; sorted by K*
            agg = (sub2.groupby(budget_col)
                       .agg(x_mean=(x_col, "mean"),
                            y_mean=(y_col, "mean"),
                            y_std=(y_col, "std"),
                            n=(y_col, "count"))
                       .reset_index()
                       .sort_values("x_mean"))
            ci = 1.96 * agg["y_std"] / np.sqrt(agg["n"].clip(lower=1))
            ax.errorbar(
                agg["x_mean"], agg["y_mean"], yerr=ci,
                marker=marker, color=color, linewidth=1.8, markersize=6,
                capsize=3, capthick=1.0, elinewidth=1.0,
                label=alloc,
            )
        ax.set_title(f"$T_{{\\max}}$ = {int(t_max)}")
        ax.set_xlabel(r"Privacy bound $K^\star$ (nats)")
        ax.set_ylabel("Test loss")
        ax.legend(loc="best", fontsize=8)

    fig.suptitle(f"Setting {setting}: Privacy-Utility Frontier", y=1.02)
    fig.tight_layout()
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
    topology_col: str = "topology.type",
) -> None:
    """Plot $K^\\star$ vs η for topology-aware vs uniform, faceted by topology.

    If the DataFrame contains multiple values in ``topology_col``, produce one
    panel per topology (left → right in canonical asymmetry order: ring, line,
    hierarchical, star) so the manuscript-grade figure shows the gap-pattern
    differs across topology classes:

    - Ring: gap = 0 everywhere (degenerate; theorem reduction case).
    - Line: gap is small (~0.02 nats), grows linearly with η.
    - Hierarchical (uneven groups): gap grows continuously with η (~0 → 0.5).
    - Star: gap is constant ~$an/U$ across η > 0 (asymptotic regime).

    Each panel also overlays a thin secondary axis showing the gap (K_uniform −
    K_topology_aware) — the headline quantity for §6 of the paper.

    Under Theorem 2 + Corollary 3, every panel's gap is non-negative and the
    η = 0 column has gap = 0 (IID null sanity check).
    """
    _set_style()
    import matplotlib.pyplot as plt

    df = df.dropna(subset=[metric_col, eta_col, allocation_col])

    # Detect topology variation. If the column is missing or all values are the
    # same, fall back to a single panel.
    if topology_col in df.columns and df[topology_col].nunique() > 1:
        # Canonical left-to-right ordering by leverage asymmetry
        canonical_order = ["ring", "line", "hierarchical", "star"]
        topologies = sorted(
            df[topology_col].unique(),
            key=lambda t: canonical_order.index(t) if t in canonical_order else 99,
        )
    else:
        topologies = [None]

    n_panels = len(topologies)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(5 * n_panels, 3.8),
        sharex=True, sharey=False,
    )
    if n_panels == 1:
        axes = [axes]

    color_map = {"topology_aware": "#1f77b4", "uniform": "#d62728"}
    marker_map = {"topology_aware": "o", "uniform": "s"}

    for ax, topo in zip(axes, topologies):
        if topo is None:
            sub_df = df
            panel_title = "η-sweep"
        else:
            sub_df = df[df[topology_col] == topo]
            panel_title = topo

        # Plot K_star and K_uniform as separate curves
        for alloc, sub in sub_df.groupby(allocation_col):
            agg = sub.groupby(eta_col)[metric_col].agg(["mean", "std", "count"]).reset_index()
            ci = 1.96 * agg["std"] / np.sqrt(agg["count"].clip(lower=1))
            ax.plot(
                agg[eta_col], agg["mean"],
                marker=marker_map.get(alloc, "o"),
                label=alloc,
                color=color_map.get(alloc, "gray"),
                linewidth=1.8, markersize=5,
            )
            ax.fill_between(
                agg[eta_col], agg["mean"] - ci, agg["mean"] + ci,
                alpha=0.2, color=color_map.get(alloc, "gray"),
            )

        # Compute and annotate the gap (K_uniform - K_topology_aware) per η
        try:
            pivot = sub_df.pivot_table(
                values=metric_col, index=eta_col, columns=allocation_col, aggfunc="mean",
            )
            if {"uniform", "topology_aware"}.issubset(pivot.columns):
                gap = pivot["uniform"] - pivot["topology_aware"]
                # Print the gap values inside the panel's lower-right corner
                gap_text = "Gap (K_uniform − K*):\n" + "\n".join(
                    f"  η={e:.2f}: {g:+.3f}" for e, g in gap.items()
                )
                ax.text(
                    0.98, 0.02, gap_text, transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=7,
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2),
                )
        except Exception:
            pass

        ax.set_title(panel_title)
        ax.set_xlabel(r"Topology-data coupling $\eta$")
        ax.legend(loc="upper left", fontsize=8)

    axes[0].set_ylabel(r"Worst-case privacy bound $K^\star$ (nats)")
    fig.suptitle(
        "Setting C: Topology-aware allocation gap across topologies",
        y=1.02,
    )
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
