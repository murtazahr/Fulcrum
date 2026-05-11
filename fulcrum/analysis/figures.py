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
    budget_col: str = "dp.utility_budget_U",
) -> None:
    """Privacy-utility figure for one setting.

    Layout (top to bottom):

    - **Privacy row** (3 panels, one per $T_{\\max}$): $U \\to K^\\star$.
      The signal axis. $K^\\star$ is analytic (1-D bisection on the budget
      equation) so it carries no seed noise; the topology-aware curve sits
      below uniform whenever leverage is non-uniform, and the vertical gap
      is the worst-case MI reduction in nats.

    - **Utility consistency table** (single full-width row): per
      $T_{\\max}$, mean accuracy under each allocation, max/mean
      $|\\Delta\\text{acc}|$ over the 18 paired configs (6 U × 3 seeds),
      direction tally (TA wins / ties / uniform wins), and a paired
      t-test p-value across all 18 pairs. Replaces an earlier per-$U$
      delta scatter that was visually weak because most deltas are zero.

    Why not plot $(K^\\star, \\text{loss})$ directly: at fixed $T_{\\max}$,
    test loss is near-flat in $K^\\star$ (loss is dominated by $T_{\\max}$
    and seed noise, not by allocation) so the strict Pareto frontier
    extracts 1-2 non-dominated points and the visual signal collapses to
    a horizontal jog dominated by seed variance. Splitting the privacy
    signal (plot) from the utility null-result (table) lets each carry
    its own evidentiary form: a line plot for the dominance claim, and
    a statistical summary for the invariance claim.
    """
    _set_style()
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator, ScalarFormatter

    needed = ["K_star", "K_uniform", "test_accuracy", "dp.allocation",
              "dp.observation_window", budget_col]
    df = df.dropna(subset=needed)
    if df.empty:
        raise ValueError(f"No data for privacy-utility plot in setting {setting}")

    t_max_values = sorted(df["dp.observation_window"].unique())
    n_panels = len(t_max_values)

    fig = plt.figure(figsize=(4.6 * n_panels, 4.8))
    gs = GridSpec(
        nrows=2, ncols=n_panels,
        height_ratios=[3.0, 1.2],
        hspace=0.55, wspace=0.30,
        figure=fig,
    )
    priv_axes = [fig.add_subplot(gs[0, c]) for c in range(n_panels)]
    table_ax = fig.add_subplot(gs[1, :])
    table_ax.axis("off")

    color_map = {"topology_aware": "#1f77b4", "uniform": "#d62728"}
    marker_map = {"topology_aware": "o", "uniform": "s"}
    alloc_order = ["uniform", "topology_aware"]
    seed_col = "experiment.seed"

    # --- Privacy panels --------------------------------------------------
    table_rows: list[list[str]] = []
    for col, t_max in enumerate(t_max_values):
        sub = df[df["dp.observation_window"] == t_max]
        ax_priv = priv_axes[col]

        means: dict[str, pd.DataFrame] = {}
        for alloc in alloc_order:
            sub2 = sub[sub["dp.allocation"] == alloc]
            if sub2.empty:
                continue
            means[alloc] = (sub2.groupby(budget_col)
                                .agg(K_mean=("K_star", "mean"))
                                .reset_index()
                                .sort_values(budget_col))

        for alloc in alloc_order:
            if alloc not in means:
                continue
            agg = means[alloc]
            ax_priv.plot(
                agg[budget_col], agg["K_mean"],
                marker=marker_map[alloc], color=color_map[alloc],
                linewidth=2.0, markersize=6,
                label=alloc, zorder=3,
            )

        if {"topology_aware", "uniform"}.issubset(means):
            ta = means["topology_aware"].set_index(budget_col)
            un = means["uniform"].set_index(budget_col)
            common = ta.index.intersection(un.index).sort_values()
            ax_priv.fill_between(
                common, ta.loc[common, "K_mean"], un.loc[common, "K_mean"],
                color="#1f77b4", alpha=0.12, label="gap", zorder=1,
            )

        # Privacy-axis cosmetics
        ax_priv.set_title(f"$T_{{\\max}}$ = {int(t_max)}")
        ax_priv.set_xscale("log")
        ax_priv.set_yscale("log")
        ax_priv.set_xlabel(r"Utility budget $U = \sum_i \sigma_i^2$")
        if col == 0:
            ax_priv.set_ylabel(r"Privacy bound $K^\star$ (nats)")
        if means:
            k_range = pd.concat([m["K_mean"] for m in means.values()])
            k_lo, k_hi = k_range.min(), k_range.max()
            nice = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70]
            yticks = [t for t in nice if k_lo / 1.2 <= t <= k_hi * 1.2]
            if len(yticks) >= 2:
                ax_priv.yaxis.set_major_locator(FixedLocator(yticks))
                ax_priv.yaxis.set_major_formatter(ScalarFormatter())
                ax_priv.yaxis.set_minor_locator(NullLocator())
        u_values = sorted(sub[budget_col].unique())
        if u_values:
            ax_priv.xaxis.set_major_locator(FixedLocator(u_values))
            ax_priv.xaxis.set_major_formatter(FuncFormatter(lambda x, _pos: f"{x:g}"))
            ax_priv.xaxis.set_minor_locator(NullLocator())
            ax_priv.tick_params(axis="x", labelsize=8)

        # --- Compute one row of the utility-consistency table ------------
        row_cells = [f"$T_{{\\max}}={int(t_max)}$"]
        if seed_col in sub.columns:
            paired = (sub.pivot_table(
                          values="test_accuracy",
                          index=[budget_col, seed_col],
                          columns="dp.allocation",
                          aggfunc="mean")
                        .dropna(how="any"))
            if {"topology_aware", "uniform"}.issubset(paired.columns):
                ta_acc = paired["topology_aware"]
                un_acc = paired["uniform"]
                delta = ta_acc - un_acc
                n_pairs = int(len(delta))

                # Paired t-test (statsmodels-free; use scipy if available)
                try:
                    from scipy.stats import ttest_rel
                    pval = float(ttest_rel(ta_acc, un_acc).pvalue)
                    pval_str = f"{pval:.3f}" if pval >= 1e-3 else f"{pval:.1e}"
                except Exception:
                    pval_str = "—"

                wins = int((delta > 0).sum())
                losses = int((delta < 0).sum())
                ties = int((delta == 0).sum())

                row_cells.extend([
                    f"{100 * ta_acc.mean():.2f}\\%",
                    f"{100 * un_acc.mean():.2f}\\%",
                    f"{100 * delta.abs().max():.2f}",
                    f"{100 * delta.abs().mean():.3f}",
                    f"{wins}/{ties}/{losses}",
                    pval_str,
                    f"{n_pairs}",
                ])
            else:
                row_cells.extend(["—"] * 7)
        else:
            row_cells.extend(["—"] * 7)
        table_rows.append(row_cells)

    # Shared legend on the leftmost privacy panel only
    priv_axes[0].legend(loc="upper right", fontsize=8, framealpha=0.85)

    # --- Render the utility-consistency table ----------------------------
    col_labels = [
        r"$T_{\max}$",
        r"TA acc (mean \%)",
        r"Uniform acc (mean \%)",
        r"$\max|\Delta|$ (pp)",
        r"$\mathrm{mean}|\Delta|$ (pp)",
        r"TA wins / ties / loses",
        r"paired $t$ $p$",
        "$n$ pairs",
    ]
    table = table_ax.table(
        cellText=table_rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.4)
    # Bold the header row
    for c in range(len(col_labels)):
        table[(0, c)].set_text_props(weight="bold")
        table[(0, c)].set_facecolor("#eef")
    table_ax.set_title(
        "Utility consistency (paired across $U$ and seed; values in percentage points)",
        fontsize=9, pad=6,
    )

    fig.suptitle(f"Setting {setting}: Privacy-Utility Trade-off", y=0.995)
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
