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
    """Privacy line plots (one panel per $T_{\\max}$) for a single setting.

    Plots $U \\to K^\\star$ for each allocation strategy, with the gap
    between topology-aware and uniform shaded. $K^\\star$ is analytic
    (1-D bisection on the budget equation) so it carries no seed noise;
    the topology-aware curve sits strictly below uniform whenever leverage
    is non-uniform, and the vertical gap is the worst-case MI reduction
    in nats.

    Utility consistency (the "defense costs no utility" half of the
    headline claim) is reported separately via
    :func:`utility_consistency_table`, which returns a pandas DataFrame
    suitable for ``df.to_latex(...)``; the manuscript places that table
    alongside this figure but as a distinct float.
    """
    _set_style()
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator, ScalarFormatter

    needed = ["K_star", "K_uniform", "dp.allocation",
              "dp.observation_window", budget_col]
    df = df.dropna(subset=needed)
    if df.empty:
        raise ValueError(f"No data for privacy plot in setting {setting}")

    t_max_values = sorted(df["dp.observation_window"].unique())
    n_panels = len(t_max_values)

    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(4.4 * n_panels, 3.4),
        sharex=False, sharey=False,
    )
    if n_panels == 1:
        axes = [axes]

    color_map = {"topology_aware": "#1f77b4", "uniform": "#d62728"}
    marker_map = {"topology_aware": "o", "uniform": "s"}
    alloc_order = ["uniform", "topology_aware"]

    for col, t_max in enumerate(t_max_values):
        sub = df[df["dp.observation_window"] == t_max]
        ax = axes[col]

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
            ax.plot(
                agg[budget_col], agg["K_mean"],
                marker=marker_map[alloc], color=color_map[alloc],
                linewidth=2.0, markersize=6,
                label=alloc, zorder=3,
            )

        if {"topology_aware", "uniform"}.issubset(means):
            ta = means["topology_aware"].set_index(budget_col)
            un = means["uniform"].set_index(budget_col)
            common = ta.index.intersection(un.index).sort_values()
            ax.fill_between(
                common, ta.loc[common, "K_mean"], un.loc[common, "K_mean"],
                color="#1f77b4", alpha=0.12, label="gap", zorder=1,
            )

        ax.set_title(f"$T_{{\\max}}$ = {int(t_max)}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Utility budget $U = \sum_i \sigma_i^2$")
        if col == 0:
            ax.set_ylabel(r"Privacy bound $K^\star$ (nats)")

        # y-axis: nice numerals on the log scale
        if means:
            k_range = pd.concat([m["K_mean"] for m in means.values()])
            k_lo, k_hi = k_range.min(), k_range.max()
            nice = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70]
            yticks = [t for t in nice if k_lo / 1.2 <= t <= k_hi * 1.2]
            if len(yticks) >= 2:
                ax.yaxis.set_major_locator(FixedLocator(yticks))
                ax.yaxis.set_major_formatter(ScalarFormatter())
                ax.yaxis.set_minor_locator(NullLocator())

        # x-axis: ticks at every sampled U value
        u_values = sorted(sub[budget_col].unique())
        if u_values:
            ax.xaxis.set_major_locator(FixedLocator(u_values))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _pos: f"{x:g}"))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.tick_params(axis="x", labelsize=8)

    # One legend in the leftmost panel
    axes[0].legend(loc="upper right", fontsize=8, framealpha=0.85)

    fig.suptitle(f"Setting {setting}: Privacy bound vs. utility budget", y=1.00)
    fig.tight_layout()
    _save(fig, output_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Utility-consistency summary table — companion to plot_pareto_setting
# ---------------------------------------------------------------------------

def utility_consistency_table(
    df: pd.DataFrame,
    budget_col: str = "dp.utility_budget_U",
    seed_col: str = "experiment.seed",
    accuracy_pp: bool = True,
) -> pd.DataFrame:
    """Build the utility-invariance table that accompanies :func:`plot_pareto_setting`.

    For each $T_{\\max}$, pairs runs at matched $(U, \\text{seed})$ across the
    two allocation strategies and reports:

    - mean test accuracy under each allocation
    - max and mean $|\\Delta\\text{acc}|$ across all paired configs
    - direction tally (TA wins / ties / uniform wins)
    - paired $t$-test $p$-value over all pairs
    - number of pairs

    Returns a tidy DataFrame indexed by $T_{\\max}$, ready for
    ``df.to_latex(...)`` or ``df.to_markdown(...)``. Accuracy values are in
    percentage points (multiply by 100) when ``accuracy_pp=True``.

    Args:
        df: completed runs DataFrame (from :func:`load_runs_df`).
        budget_col: column holding utility budget $U$.
        seed_col: column holding the random seed (for pairing).
        accuracy_pp: if True, report accuracy fields ×100 (percentage points).

    Returns:
        DataFrame with one row per $T_{\\max}$ and columns:
        ``ta_acc_mean``, ``unif_acc_mean``, ``abs_delta_max``,
        ``abs_delta_mean``, ``ta_wins``, ``ties``, ``unif_wins``,
        ``paired_t_pvalue``, ``n_pairs``.
    """
    needed = ["test_accuracy", "dp.allocation", "dp.observation_window",
              budget_col, seed_col]
    df = df.dropna(subset=needed)

    try:
        from scipy.stats import ttest_rel  # noqa: F401  (used per-row below)
        have_scipy = True
    except Exception:
        have_scipy = False

    rows: list[dict] = []
    scale = 100.0 if accuracy_pp else 1.0
    for t_max in sorted(df["dp.observation_window"].unique()):
        sub = df[df["dp.observation_window"] == t_max]
        paired = (sub.pivot_table(
                      values="test_accuracy",
                      index=[budget_col, seed_col],
                      columns="dp.allocation",
                      aggfunc="mean")
                    .dropna(how="any"))
        if not {"topology_aware", "uniform"}.issubset(paired.columns):
            continue
        ta = paired["topology_aware"]
        un = paired["uniform"]
        delta = ta - un

        if have_scipy and len(delta) > 1:
            from scipy.stats import ttest_rel
            pval: float | None = float(ttest_rel(ta, un).pvalue)
        else:
            pval = None

        rows.append({
            "T_max": int(t_max),
            "ta_acc_mean": scale * float(ta.mean()),
            "unif_acc_mean": scale * float(un.mean()),
            "abs_delta_max": scale * float(delta.abs().max()),
            "abs_delta_mean": scale * float(delta.abs().mean()),
            "ta_wins": int((delta > 0).sum()),
            "ties": int((delta == 0).sum()),
            "unif_wins": int((delta < 0).sum()),
            "paired_t_pvalue": pval,
            "n_pairs": int(len(delta)),
        })
    return pd.DataFrame(rows).set_index("T_max")


def utility_consistency_latex(
    table_df: pd.DataFrame,
    setting: str,
    caption: str | None = None,
    label: str | None = None,
) -> str:
    """Format :func:`utility_consistency_table` output as a manuscript-ready LaTeX
    ``booktabs`` table.

    Returns the LaTeX string; caller is responsible for writing it to disk
    or piping it into the manuscript.

    Designed to compile under the project's existing ``\\usepackage{booktabs}``
    (already in ``paper/main.tex``).
    """
    label = label or f"tab:util-consistency-{setting.lower()}"
    caption = caption or (
        f"Setting {setting}: utility consistency under topology-aware vs.\\ "
        f"uniform DP-SGD allocation. Each row aggregates paired runs at "
        f"matched $(U, \\text{{seed}})$. Accuracy fields in percentage points; "
        f"$p$ from a paired $t$-test across all pairs."
    )

    def fmt_p(p):
        if p is None or pd.isna(p):
            return "---"
        return f"{p:.3f}" if p >= 1e-3 else f"{p:.1e}"

    body = []
    for t_max, row in table_df.iterrows():
        body.append(" & ".join([
            f"{t_max}",
            f"{row['ta_acc_mean']:.2f}",
            f"{row['unif_acc_mean']:.2f}",
            f"{row['abs_delta_max']:.2f}",
            f"{row['abs_delta_mean']:.3f}",
            f"{int(row['ta_wins'])}/{int(row['ties'])}/{int(row['unif_wins'])}",
            fmt_p(row["paired_t_pvalue"]),
            f"{int(row['n_pairs'])}",
        ]) + r" \\")

    return "\n".join([
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{@{}rcccccccc@{}}",
        r"\toprule",
        r"$T_{\max}$ & TA acc (\%) & Uniform acc (\%) & "
        r"$\max|\Delta|$ (pp) & $\overline{|\Delta|}$ (pp) & "
        r"TA / tie / Unif & paired $t$ $p$ & $n$ \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])


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

# ---------------------------------------------------------------------------
# Attack-lift figures (companions to channel ablation)
# ---------------------------------------------------------------------------

CHANNEL_ORDER = ["A1", "A2_topo", "A2_org", "A2_full"]
CHANNEL_COLOR = {
    "A1":      "#888888",   # parameter-only — neutral grey
    "A2_topo": "#2ca02c",   # structural — green
    "A2_org":  "#d62728",   # organisational — red (the dominant channel in Setting C)
    "A2_full": "#1f77b4",   # combined — blue
}
CHANNEL_MARKER = {"A1": "o", "A2_topo": "^", "A2_org": "s", "A2_full": "D"}


def plot_attack_lift_eta(
    df: pd.DataFrame,
    setting: str,
    output_path: str | Path,
    eta_col: str = "data.eta",
    lift_col: str = "attack_lift",
    channel_col: str = "channel",
) -> None:
    """Attack lift vs $\\eta$, one curve per channel ablation.

    Expects ``df`` to be the (target_run, channel) attack-evaluation table
    joined with the target run's config (so ``data.eta`` is available).
    Plots the mean attack lift at each $\\eta$ with a shaded 95\\% CI from
    seed variance, with a horizontal zero reference line.

    The IID-null sanity is visible directly: at $\\eta=0$ every channel
    should sit at or below zero.
    """
    _set_style()
    import matplotlib.pyplot as plt

    df = df.dropna(subset=[eta_col, lift_col, channel_col])
    if df.empty:
        raise ValueError(f"No attack-lift data for setting {setting}")

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--", zorder=1)

    for channel in CHANNEL_ORDER:
        sub = df[df[channel_col] == channel]
        if sub.empty:
            continue
        agg = (sub.groupby(eta_col)[lift_col]
                  .agg(["mean", "std", "count"])
                  .reset_index()
                  .sort_values(eta_col))
        ci = 1.96 * agg["std"].fillna(0.0) / np.sqrt(agg["count"].clip(lower=1))
        ax.plot(
            agg[eta_col], agg["mean"],
            marker=CHANNEL_MARKER.get(channel, "o"),
            color=CHANNEL_COLOR.get(channel, "black"),
            linewidth=1.8, markersize=6,
            label=channel, zorder=3,
        )
        ax.fill_between(
            agg[eta_col],
            agg["mean"] - ci, agg["mean"] + ci,
            color=CHANNEL_COLOR.get(channel, "black"),
            alpha=0.15, zorder=2,
        )

    ax.set_xlabel(r"Topology-data coupling $\eta$")
    ax.set_ylabel("Attack lift (constant-mean baseline − calibration loss)")
    ax.set_title(f"Setting {setting}: TADI attack lift across channel ablations")
    ax.legend(loc="best", fontsize=8, framealpha=0.85)
    fig.tight_layout()
    _save(fig, output_path)
    plt.close(fig)


def plot_attack_lift_vs_K(
    df: pd.DataFrame,
    setting: str,
    output_path: str | Path,
    k_col: str = "K_star",
    lift_col: str = "attack_lift",
    channel_col: str = "channel",
    eta_col: str = "data.eta",
) -> None:
    """Attack lift against the predicted privacy bound $K^\\star$.

    One panel per channel; each panel scatters every target run as a
    point coloured by $\\eta$ (deeper colour = stronger coupling).
    Annotates the Pearson correlation and a linear best-fit line.
    Reports correlation in the panel title.

    Theorem 1 predicts higher $K^\\star$ → higher achievable adversary
    capability. The empirical correlation tests how predictive the
    theoretical bound is. For Setting C the correlation is +0.35 for
    the org and full channels (modest but directional).
    """
    _set_style()
    import matplotlib.pyplot as plt
    from matplotlib import cm

    df = df.dropna(subset=[k_col, lift_col, channel_col])
    if df.empty:
        raise ValueError(f"No K* / attack-lift data for setting {setting}")

    n_channels = len(CHANNEL_ORDER)
    fig, axes = plt.subplots(
        1, n_channels, figsize=(3.0 * n_channels, 3.4),
        sharey=True,
    )
    if n_channels == 1:
        axes = [axes]

    eta_norm = None
    if eta_col in df.columns:
        eta_vals = df[eta_col].dropna()
        if not eta_vals.empty:
            eta_norm = plt.Normalize(vmin=eta_vals.min(), vmax=eta_vals.max())
    cmap = cm.get_cmap("viridis")

    for ax, channel in zip(axes, CHANNEL_ORDER):
        sub = df[df[channel_col] == channel].copy()
        if sub.empty:
            ax.set_title(f"{channel}\n(no data)")
            continue
        if eta_norm is not None and eta_col in sub.columns:
            colors = cmap(eta_norm(sub[eta_col].fillna(eta_norm.vmin)))
        else:
            colors = CHANNEL_COLOR.get(channel, "black")
        ax.scatter(sub[k_col], sub[lift_col], c=colors, alpha=0.7, s=22,
                   edgecolors="none", zorder=3)

        # Linear fit + correlation
        if len(sub) >= 2:
            corr = float(sub[[k_col, lift_col]].corr().iloc[0, 1])
            xs = np.linspace(sub[k_col].min(), sub[k_col].max(), 50)
            # Avoid polyfit on degenerate variance
            if sub[k_col].std() > 1e-12:
                a, b = np.polyfit(sub[k_col], sub[lift_col], 1)
                ax.plot(xs, a * xs + b, color="black", linewidth=1.0,
                        linestyle="--", zorder=2)
            title = f"{channel}\n$r$ = {corr:+.3f}  ($n$ = {len(sub)})"
        else:
            title = f"{channel}\n(n={len(sub)})"

        ax.axhline(0.0, color="gray", linewidth=0.6, linestyle=":", zorder=1)
        ax.set_xscale("log")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(r"Predicted bound $K^\star$ (nats)")

    axes[0].set_ylabel("Attack lift")

    # Colourbar for η
    if eta_norm is not None:
        sm = cm.ScalarMappable(norm=eta_norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, orientation="vertical",
                            shrink=0.7, pad=0.02, aspect=20)
        cbar.set_label(r"$\eta$", rotation=0, labelpad=8)

    fig.suptitle(
        f"Setting {setting}: TADI attack lift vs. theoretical bound",
        y=1.02,
    )
    _save(fig, output_path)
    plt.close(fig)


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
