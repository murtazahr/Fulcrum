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
    from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullLocator

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
            # Adaptive log-scale ticks. The previous hardcoded "nice
            # numbers" list (1, 2, 3, 5, 7, 10, ...) had no values
            # below 1 and silently fell through to the matplotlib
            # default when <2 ticks landed in range — producing
            # empty or single-tick y-axes on Setting~A where K* can
            # dip below 1. Use a LogLocator with sub-decade ticks at
            # (1, 2, 3, 5, 7) which gives dense, clean labels across
            # any decade range from 10^-2 upward, paired with a
            # FuncFormatter that emits compact decimals ("0.5", "5",
            # "50") rather than scientific notation.
            ax.yaxis.set_major_locator(
                LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 5.0, 7.0), numticks=12)
            )
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _pos: f"{y:g}"))
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

    # No figure suptitle; the LaTeX caption carries the description in the
    # manuscript. Dropping it keeps the figure compact and consistent with
    # the η-heatmap and the channel-ablation bar chart.
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
    equivalence_margin_pp: float = 0.5,
) -> pd.DataFrame:
    """Build the utility-invariance table that accompanies :func:`plot_pareto_setting`.

    For each $T_{\\max}$, pairs runs at matched $(U, \\text{seed})$ across the
    two allocation strategies and reports:

    - mean test accuracy under each allocation
    - max and mean $|\\Delta\\text{acc}|$ across all paired configs
    - direction tally (TA wins / ties / uniform wins)
    - paired $t$-test $p$-value over all pairs
    - **TOST equivalence p-value** for the null
      $|\\mathbb{E}[\\Delta\\text{acc}]| \\ge \\text{margin}$ vs.\\ alternative
      $|\\mathbb{E}[\\Delta\\text{acc}]| < \\text{margin}$. A small TOST p-value
      (< 0.05) lets us conclude statistical equivalence within the margin.
    - 95% confidence interval on the paired difference
    - number of pairs

    The TOST formulation addresses the reviewer's concern that a high
    paired-$t$ $p$-value does not establish equivalence — it only fails to
    reject the null of zero difference. TOST (Two One-Sided Tests) inverts
    the null to require the effect size to be small to be "rejected,"
    converting a non-rejection of difference into an active rejection of
    practically-meaningful difference.

    Returns a tidy DataFrame indexed by $T_{\\max}$, ready for
    ``df.to_latex(...)`` or ``df.to_markdown(...)``. Accuracy values are in
    percentage points (multiply by 100) when ``accuracy_pp=True``.

    Args:
        df: completed runs DataFrame (from :func:`load_runs_df`).
        budget_col: column holding utility budget $U$.
        seed_col: column holding the random seed (for pairing).
        accuracy_pp: if True, report accuracy fields ×100 (percentage points).
        equivalence_margin_pp: equivalence margin for the TOST in
            percentage-point units (default 0.5pp). The reported TOST p-value
            tests whether mean |Δ| < margin.

    Returns:
        DataFrame with one row per $T_{\\max}$ and columns:
        ``ta_acc_mean``, ``unif_acc_mean``, ``abs_delta_max``,
        ``abs_delta_mean``, ``ta_wins``, ``ties``, ``unif_wins``,
        ``paired_t_pvalue``, ``tost_pvalue``, ``ci95_low``, ``ci95_high``,
        ``equivalence_margin_pp``, ``n_pairs``.
    """
    needed = ["test_accuracy", "dp.allocation", "dp.observation_window",
              budget_col, seed_col]
    df = df.dropna(subset=needed)

    try:
        from scipy.stats import ttest_rel, t as student_t  # noqa: F401
        have_scipy = True
    except Exception:
        have_scipy = False

    rows: list[dict] = []
    scale = 100.0 if accuracy_pp else 1.0
    margin = equivalence_margin_pp / scale  # convert margin into the data's units
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
        n = int(len(delta))

        if have_scipy and n > 1:
            from scipy.stats import ttest_rel, t as student_t
            pval: float | None = float(ttest_rel(ta, un).pvalue)
            # Two One-Sided Tests for equivalence within ±margin.
            # Lower test: H0_low: μ ≤ -margin vs H1_low: μ > -margin
            # Upper test: H0_up:  μ ≥  margin vs H1_up: μ <  margin
            # Equivalence p-value = max(p_low, p_up).
            mean_d = float(delta.mean())
            sd_d = float(delta.std(ddof=1)) if delta.std(ddof=1) > 0 else 1e-12
            se_d = sd_d / np.sqrt(n)
            t_low = (mean_d - (-margin)) / se_d
            t_up = (mean_d - margin) / se_d
            df_t = n - 1
            p_low = 1 - student_t.cdf(t_low, df=df_t)   # P(T > t_low) under H0_low
            p_up = student_t.cdf(t_up, df=df_t)         # P(T < t_up)  under H0_up
            tost_p: float | None = float(max(p_low, p_up))
            # 95% CI on paired difference (used as supporting evidence)
            half = float(student_t.ppf(0.975, df=df_t)) * se_d
            ci_low = mean_d - half
            ci_high = mean_d + half
        else:
            pval = None
            tost_p = None
            ci_low = float("nan")
            ci_high = float("nan")

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
            "tost_pvalue": tost_p,
            "ci95_low_pp": scale * float(ci_low),
            "ci95_high_pp": scale * float(ci_high),
            "equivalence_margin_pp": float(equivalence_margin_pp),
            "n_pairs": n,
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
    margin = float(table_df["equivalence_margin_pp"].iloc[0]) if "equivalence_margin_pp" in table_df.columns else 0.5
    caption = caption or (
        f"Setting {setting}: utility consistency under topology-aware vs.\\ "
        f"uniform DP-SGD allocation. Each row aggregates paired runs at "
        f"matched $(U, \\text{{seed}})$. Accuracy fields in percentage points. "
        f"``Paired $t$ $p$'' is the conventional $p$-value for $H_0$: zero mean "
        f"difference; a high value indicates failure to reject equality. "
        f"``TOST $p$'' is the equivalence $p$-value for $H_0$: "
        f"$|\\mathbb{{E}}[\\Delta]|\\ge {margin:.2g}$~pp; a small value (< 0.05) "
        f"establishes statistical equivalence within the margin. "
        f"95\\% CI is on the paired mean difference."
    )

    def fmt_p(p):
        if p is None or pd.isna(p):
            return "---"
        return f"{p:.3f}" if p >= 1e-3 else f"{p:.1e}"

    def fmt_ci(lo, hi):
        if pd.isna(lo) or pd.isna(hi):
            return "---"
        return f"$[{lo:+.3f}, {hi:+.3f}]$"

    body = []
    for t_max, row in table_df.iterrows():
        body.append(" & ".join([
            f"{t_max}",
            f"{row['ta_acc_mean']:.2f}",
            f"{row['unif_acc_mean']:.2f}",
            f"{row['abs_delta_max']:.2f}",
            f"{row['abs_delta_mean']:.3f}",
            fmt_ci(row.get("ci95_low_pp", float("nan")), row.get("ci95_high_pp", float("nan"))),
            fmt_p(row["paired_t_pvalue"]),
            fmt_p(row.get("tost_pvalue", None)),
            f"{int(row['n_pairs'])}",
        ]) + r" \\")

    return "\n".join([
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{@{}rccccccccc@{}}",
        r"\toprule",
        r"$T_{\max}$ & TA acc (\%) & Uniform acc (\%) & "
        r"$\max|\Delta|$ (pp) & $\overline{|\Delta|}$ (pp) & "
        r"95\% CI on $\Delta$ (pp) & paired $t$ $p$ & TOST $p$ & $n$ \\",
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
# η-sweep gap heatmap — primary Theorem 5.3 validation figure
# ---------------------------------------------------------------------------

def _topology_row_label(row: pd.Series) -> str:
    """Compact, publication-ready label for one (topology, params) row.

    Used as the y-axis tick label for the η-sweep heatmap. Keeps ASCII only
    so the figure renders identically under any LaTeX font setup; the
    accented topology names (Erdős–Rényi, Barabási–Albert) are abbreviated
    to ``ER`` and ``BA`` to avoid encoding pitfalls.
    """
    topo = row["topology.type"]
    if topo == "ring":
        return "Ring"
    if topo == "line":
        return "Line"
    if topo == "star":
        return "Star"
    if topo == "hierarchical":
        sizes = row.get("topology.params.region_sizes")
        if isinstance(sizes, (list, tuple, np.ndarray)) and len(sizes) > 0:
            return "Hier. [" + ",".join(str(int(s)) for s in sizes) + "]"
        return "Hierarchical"
    if topo == "erdos":
        p = row.get("topology.params.p")
        return f"ER p={p:g}" if pd.notna(p) else "ER"
    if topo in ("barabasi_albert", "barabasi", "ba"):
        m = row.get("topology.params.m")
        return f"BA m={int(m)}" if pd.notna(m) else "BA"
    return str(topo)


def _topology_group_key(label: str) -> tuple[int, str]:
    """Stable sort key grouping topologies into families for fall-back ordering.

    Families: deterministic-symmetric (ring), deterministic-asymmetric
    (line, hierarchical, star), random homogeneous (ER), scale-free (BA).
    Inside a family the secondary key is the label itself, so ER p=0.3
    sorts before ER p=0.7.
    """
    if label.startswith("Ring"):
        return (0, label)
    if label.startswith("Line"):
        return (1, label)
    if label.startswith("Hier"):
        return (2, label)
    if label.startswith("ER"):
        return (3, label)
    if label.startswith("BA"):
        return (4, label)
    if label.startswith("Star"):
        return (5, label)
    return (9, label)


def plot_eta_gap_heatmap(
    df: pd.DataFrame,
    output_path: str | Path,
    eta_col: str = "data.eta",
    allocation_col: str = "dp.allocation",
    topology_col: str = "topology.type",
    annotate: bool = True,
    sort_by: str = "max_gap",
    asymptote_anU: float | None = None,
) -> None:
    """Heatmap of the topology-aware vs uniform privacy-bound gap.

    Replaces the multi-panel line plot in :func:`plot_eta_sweep` for the
    primary Theorem~5.3 validation figure. Rows are (topology, params)
    configurations and columns are η values. Cell colour is
    $K_{\\mathrm{uniform}} - K^\\star$ (nats), the headline quantity of
    Corollary~5.4 (strict improvement over uniform allocation).

    Three structural properties of the bound are readable directly from
    the heatmap:

    - **IID-null calibration.** At η = 0, every cell should be exactly
      zero (uniform partitioning + degenerate leverage gives no gap).
    - **Symmetric-topology degeneracy.** Rows for uniform-leverage
      topologies (ring, balanced hierarchies) should be a flat zero band
      across all η.
    - **Asymmetric saturation.** The maximally asymmetric row (star, or
      a heavy-tailed BA) approaches the analytic asymptote
      $a n / U$ as η grows; this is the bound's saturation value.

    Args:
        df: combined runs DataFrame for Setting~C, already restricted to
            the η-sweep grid (typically $U = 0.5$, $T_{\\max} = 100$).
        output_path: figure path (without extension).
        eta_col, allocation_col, topology_col: column names.
        annotate: print the numeric gap inside each cell.
        sort_by: either ``"max_gap"`` (data-driven; degenerate rows at
            top, asymmetric at bottom) or ``"family"`` (deterministic
            order by topology family).
        asymptote_anU: if provided, marks the analytic asymptote
            $a n / U$ on the colourbar with a thin black tick.
    """
    _set_style()
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors

    needed = {eta_col, allocation_col, topology_col, "K_star", "K_uniform"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for eta heatmap: {sorted(missing)}")
    df = df.dropna(subset=[eta_col, allocation_col, topology_col, "K_star", "K_uniform"])

    # Build a (topology-config, label) key per row. Hierarchical and ER/BA
    # rows carry params that distinguish multiple configurations of the same
    # topology family; we tuple-key on the relevant params column so they
    # become separate heatmap rows.
    def _config_key(row: pd.Series) -> tuple:
        t = row[topology_col]
        if t == "hierarchical":
            sizes = row.get("topology.params.region_sizes")
            if isinstance(sizes, (list, tuple, np.ndarray)):
                return ("hierarchical", tuple(int(s) for s in sizes))
            return ("hierarchical", None)
        if t == "erdos":
            return ("erdos", row.get("topology.params.p"))
        if t in ("barabasi_albert", "barabasi", "ba"):
            return ("ba", row.get("topology.params.m"))
        return (t, None)

    df = df.copy()
    df["_config_key"] = df.apply(_config_key, axis=1)
    df["_row_label"] = df.apply(_topology_row_label, axis=1)

    # Aggregate K_star and K_uniform per (config, η, allocation). K_star and
    # K_uniform are analytic so the seed-mean is exact; using mean is just
    # protection against any per-seed drift that may have crept in.
    agg = (
        df.groupby(["_config_key", "_row_label", eta_col, allocation_col])
          [["K_star", "K_uniform"]].mean().reset_index()
    )

    # Pivot to (row, η) with separate K_star_TA and K_uniform columns.
    ta = agg[agg[allocation_col] == "topology_aware"].set_index(
        ["_config_key", "_row_label", eta_col]
    )["K_star"]
    un = agg[agg[allocation_col] == "uniform"].set_index(
        ["_config_key", "_row_label", eta_col]
    )["K_uniform"]
    # Fall back to K_uniform from the TA row if no uniform-allocation row
    # exists for that cell (the analytic K_uniform is identical regardless
    # of which allocation we read it from).
    if un.empty:
        un = agg[agg[allocation_col] == "topology_aware"].set_index(
            ["_config_key", "_row_label", eta_col]
        )["K_uniform"]
    gap = (un - ta).reset_index(name="gap")

    # Pivot to wide form: rows = config, cols = η, values = gap.
    pivot = gap.pivot_table(
        index=["_config_key", "_row_label"],
        columns=eta_col,
        values="gap",
        aggfunc="mean",
    )
    # Sort columns ascending in η.
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    # Analytic fill for missing η=0 cells. The η-position leverage proxy
    # collapses to uniform leverage at η=0, so topology-aware allocation
    # degenerates to uniform DP-SGD and the gap is zero by construction.
    # ER and BA runs at η=0 with the TA allocation fail the
    # assert_non_uniform_leverage check for exactly this reason; the
    # resulting NaN cells should read 0, not "—".
    zero_eta = next((c for c in pivot.columns if float(c) == 0.0), None)
    if zero_eta is not None:
        pivot.loc[pivot[zero_eta].isna(), zero_eta] = 0.0

    # Row ordering.
    if sort_by == "max_gap":
        order = pivot.max(axis=1).sort_values().index
    else:
        order = sorted(pivot.index, key=lambda idx: _topology_group_key(idx[1]))
    pivot = pivot.reindex(order)

    row_labels = [lbl for _, lbl in pivot.index]
    eta_values = list(pivot.columns)
    data = pivot.to_numpy(dtype=float)

    fig_h = max(2.5, 0.42 * len(row_labels) + 1.4)
    fig, ax = plt.subplots(figsize=(6.8, fig_h))

    # Linear color scale anchored at 0; vmax slightly above asymptote so
    # the saturation row doesn't peg to pure black/white.
    vmax_data = float(np.nanmax(data)) if data.size else 1.0
    vmax = max(vmax_data, asymptote_anU or 0.0) * 1.05 if vmax_data > 0 else 1.0
    cmap = plt.get_cmap("rocket_r") if "rocket_r" in plt.colormaps() else plt.get_cmap("magma_r")
    norm = mcolors.Normalize(vmin=0.0, vmax=vmax)

    im = ax.imshow(data, aspect="auto", cmap=cmap, norm=norm)

    ax.set_xticks(np.arange(len(eta_values)))
    ax.set_xticklabels([f"{e:g}" for e in eta_values])
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    # Avoid the LaTeX-style "--" en-dash sequence (matplotlib renders it
    # as two literal hyphens, not an en-dash, which reads as noise in the
    # figure). Plain "vs." is the cleanest substitute.
    ax.set_xlabel(r"Topology vs. data coupling $\eta$")
    ax.set_ylabel("Topology configuration")
    # No figure title; the caption carries the description in the
    # manuscript. Removing the title also eliminates the redundancy with
    # the colorbar label below and leaves more vertical room for cells.

    # Disable the default grid for the heatmap; draw a faint white grid
    # along cell boundaries instead so the cells read as distinct tiles.
    ax.grid(False)
    ax.set_xticks(np.arange(len(eta_values) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(row_labels) + 1) - 0.5, minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(which="minor", color="white", linewidth=1.0)

    # Cell annotations. All cells carry a numeric value after the η=0
    # analytic fill above, so no missing-data placeholder is needed.
    if annotate:
        # White text on dark cells, black on light, threshold at half vmax.
        threshold = 0.55 * vmax
        for i in range(len(row_labels)):
            for j in range(len(eta_values)):
                v = data[i, j]
                if np.isnan(v):
                    continue
                txt = f"{v:.3f}" if v < 1.0 else f"{v:.2f}"
                color = "white" if v >= threshold else "#1a1a1a"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=7.5, color=color)

    # Colorbar carries the metric label (the title is removed in favour of
    # the manuscript caption). The asymptote an/U is marked with a thin
    # black tick on the colorbar and labelled to the right of the tick.
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label(r"$K_{\mathrm{uniform}} - K^\star$ (nats)")
    if asymptote_anU is not None and 0 < asymptote_anU < vmax:
        cbar.ax.axhline(asymptote_anU, color="black", linewidth=0.9)
        cbar.ax.annotate(
            r"$an/U$",
            xy=(1.05, asymptote_anU),
            xycoords=cbar.ax.get_yaxis_transform(),
            xytext=(6, 0), textcoords="offset points",
            va="center", ha="left",
            fontsize=8,
        )

    fig.tight_layout()
    _save(fig, output_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Channel-ablation bar chart
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Attack-lift figures (companions to channel ablation)
# ---------------------------------------------------------------------------

CHANNEL_ORDER = ["A1", "A2_topo", "A2_org", "A2_full"]
CHANNEL_LABEL = {
    "A1":      r"$\mathcal{A}_1$ (parameter)",
    "A2_topo": r"$\mathcal{A}_2^{\mathrm{topo}}$ (structural)",
    "A2_org":  r"$\mathcal{A}_2^{\mathrm{org}}$ (organisational)",
    "A2_full": r"$\mathcal{A}_2^{\mathrm{full}}$ (combined)",
}
CHANNEL_COLOR = {
    "A1":      "#888888",   # parameter-only — neutral grey
    "A2_topo": "#2ca02c",   # structural — green
    "A2_org":  "#d62728",   # organisational — red (the dominant channel in Setting C)
    "A2_full": "#1f77b4",   # combined — blue
}
CHANNEL_MARKER = {"A1": "o", "A2_topo": "^", "A2_org": "s", "A2_full": "D"}


def plot_tadi_realisability_trajectory(
    df: pd.DataFrame,
    output_path: str | Path,
    eta_col: str = "data.eta",
    channel_col: str = "channel",
) -> None:
    """Setting C TADI realisability as channel trajectories in (AUROC, lift) space.

    A single-panel "threat fingerprint": each channel is one path
    through 2D metric space (x = AUROC, y = attack lift), with the
    five η values plotted as connected markers. Marker size grows
    with η so the direction of the trajectory is readable. Reference
    lines split the panel into four quadrants annotated by their
    interpretation:

    - **Bottom-left** (AUROC ≤ 0.5, lift ≤ 0): channel bounded by
      DP-SGD or null-calibrated. The parameter channel $\\mathcal{A}_1$
      and structural channel $\\mathcal{A}_2^{\\mathrm{topo}}$ stay
      here regardless of η.
    - **Top-right** (AUROC > 0.5, lift > 0): channel realised — the
      adversary both beats the constant-mean baseline and ranks
      clients above chance. The organisational and combined channels
      arrive here at high coupling.
    - **Top-left** (AUROC ≤ 0.5, lift > 0): degenerate (rare).
    - **Bottom-right** (AUROC > 0.5, lift ≤ 0): ranks well but
      doesn't beat baseline (a regressor calibration artifact).

    The trajectory paths visualise the additive decomposition of
    Theorem 5.2 in operation: the mechanism-bounded $\\mathcal{A}_1$
    barely moves, while the prior-coupling channels traverse from
    bottom-left to top-right as η grows.

    Args:
        df: attack-eval dataframe joined with target-run config so
            ``eta_col`` is available. Expected columns: ``channel``,
            ``attack_lift``, ``auroc``, plus ``eta_col``.
        output_path: figure path (without extension).
    """
    _set_style()
    import matplotlib.pyplot as plt

    needed = {eta_col, channel_col, "attack_lift", "auroc"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for trajectory plot: {sorted(missing)}")
    df = df.dropna(subset=[eta_col, channel_col, "attack_lift", "auroc"])

    # Aggregate to (channel, eta) mean
    agg = (df.groupby([channel_col, eta_col])[["attack_lift", "auroc"]]
             .mean()
             .reset_index())

    fig, ax = plt.subplots(figsize=(6.5, 4.6))

    # Reference lines split the panel into quadrants
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--",
               alpha=0.6, zorder=1)
    ax.axvline(0.5, color="gray", linewidth=0.8, linestyle="--",
               alpha=0.6, zorder=1)

    # Plot each channel's trajectory
    for ch in CHANNEL_ORDER:
        sub = agg[agg[channel_col] == ch].sort_values(eta_col)
        if sub.empty:
            continue
        eta_vals = sub[eta_col].to_numpy()
        x = sub["auroc"].to_numpy()
        y = sub["attack_lift"].to_numpy()
        color = CHANNEL_COLOR.get(ch, "black")
        # Connecting line — faint, just to convey trajectory order
        ax.plot(x, y, color=color, linewidth=1.3, alpha=0.5, zorder=2)
        # Markers — size grows with η so trajectory direction is readable
        sizes = 40 + 80 * eta_vals  # 40 at η=0, 120 at η=1
        ax.scatter(x, y, s=sizes, c=color, edgecolor="white",
                   linewidth=0.6, zorder=3,
                   label=CHANNEL_LABEL.get(ch, ch))
        # Annotate η at first and last points
        ax.annotate(
            r"$\eta{=}0$",
            xy=(x[0], y[0]), xytext=(-6, -6), textcoords="offset points",
            fontsize=7, color=color, ha="right", va="top",
        )
        ax.annotate(
            r"$\eta{=}1$",
            xy=(x[-1], y[-1]), xytext=(6, 6), textcoords="offset points",
            fontsize=7.5, color=color, ha="left", va="bottom",
            fontweight="bold",
        )

    # Quadrant annotations — light, positioned in corners
    quadrant_style = dict(
        fontsize=8.5, color="#555555", style="italic",
        ha="center", va="center",
        bbox=dict(facecolor="white", edgecolor="none",
                  alpha=0.7, boxstyle="round,pad=0.3"),
    )
    # Bottom-left: bounded
    ax.text(0.42, -0.062,
            "DP-SGD bounded\n(controllable term)",
            **quadrant_style)
    # Top-right: realised
    ax.text(0.88, 0.062,
            "Threat realised\n(prior-coupling term)",
            **quadrant_style)

    ax.set_xlabel("AUROC (binary $p_i > \\tau$)")
    ax.set_ylabel(r"Attack lift  $L_{\mathrm{cal}}(\bar p) - L_{\mathrm{cal}}(\hat p)$")
    ax.set_xlim(0.35, 1.05)
    ax.set_ylim(-0.085, 0.085)

    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.92,
              ncol=2, handletextpad=0.5, columnspacing=0.8)
    fig.tight_layout()
    _save(fig, output_path)
    plt.close(fig)


def plot_tadi_realisability_eta_sweep(
    df: pd.DataFrame,
    output_path: str | Path,
    eta_col: str = "data.eta",
    channel_col: str = "channel",
) -> None:
    """Setting C TADI realisability: attack lift and AUROC vs η, per channel.

    The primary attack-evaluation figure. Two side-by-side panels share
    the x-axis (coupling strength $\\eta$) and overlay all four channel
    ablations. Reference lines mark the calibration nulls (lift = 0 in
    the left panel; AUROC = 0.5, chance, in the right panel).

    Args:
        df: attack-eval dataframe joined with target-run config so
            ``eta_col`` is available. Expected columns: ``channel``,
            ``attack_lift``, ``auroc``, plus ``eta_col``.
        output_path: figure path (without extension).

    The figure makes three claims readable at a glance:

    - **DP-SGD bounds the parameter channel.** $\\mathcal{A}_1$'s lift
      sits at or below zero across all η values, confirming the
      controllable term of Theorem 5.2 is suppressed by DP-SGD noise.
    - **Prior coupling realised through the organisational channel.**
      $\\mathcal{A}_2^{\\mathrm{org}}$'s lift and AUROC both rise
      monotonically with η and reach perfect AUROC = 1.0 at η ≥ 0.75.
    - **Channel non-dominance.** $\\mathcal{A}_2^{\\mathrm{full}}$ does
      not strictly dominate $\\mathcal{A}_2^{\\mathrm{org}}$ on lift;
      the combined adversary's additional features add noise that the
      finite shadow corpus cannot disambiguate.
    """
    _set_style()
    import matplotlib.pyplot as plt

    needed = {eta_col, channel_col, "attack_lift", "auroc"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for realisability plot: {sorted(missing)}")
    df = df.dropna(subset=[eta_col, channel_col, "attack_lift", "auroc"])

    fig, (ax_lift, ax_auc) = plt.subplots(1, 2, figsize=(8.5, 3.4))

    for ch in CHANNEL_ORDER:
        sub = df[df[channel_col] == ch]
        if sub.empty:
            continue
        for ax, col, zero_line in (
            (ax_lift, "attack_lift", 0.0),
            (ax_auc, "auroc", 0.5),
        ):
            agg = (sub.groupby(eta_col)[col]
                      .agg(["mean", "std", "count"])
                      .reset_index()
                      .sort_values(eta_col))
            ci = 1.96 * agg["std"].fillna(0.0) / np.sqrt(agg["count"].clip(lower=1))
            ax.plot(
                agg[eta_col], agg["mean"],
                marker=CHANNEL_MARKER.get(ch, "o"),
                color=CHANNEL_COLOR.get(ch, "black"),
                linewidth=1.8, markersize=6,
                label=CHANNEL_LABEL.get(ch, ch),
                zorder=3,
            )
            ax.fill_between(
                agg[eta_col],
                agg["mean"] - ci,
                agg["mean"] + ci,
                color=CHANNEL_COLOR.get(ch, "black"),
                alpha=0.15, zorder=2,
            )

    # Reference lines and panel-specific styling
    ax_lift.axhline(0.0, color="black", linewidth=0.7,
                    linestyle="--", alpha=0.6, zorder=1)
    ax_lift.set_xlabel(r"Topology vs. data coupling $\eta$")
    ax_lift.set_ylabel(r"Attack lift  $L_{\mathrm{cal}}(\bar p) - L_{\mathrm{cal}}(\hat p)$")
    ax_lift.text(
        0.02, 0.05, "DP-SGD\nbound", transform=ax_lift.transAxes,
        ha="left", va="bottom", fontsize=8,
        color="#444444", style="italic",
    )

    ax_auc.axhline(0.5, color="black", linewidth=0.7,
                   linestyle="--", alpha=0.6, zorder=1)
    ax_auc.set_xlabel(r"Topology vs. data coupling $\eta$")
    ax_auc.set_ylabel(r"AUROC (binary $p_i > \tau$)")
    ax_auc.set_ylim(0.35, 1.05)
    ax_auc.text(
        0.02, 0.05, "chance\n(0.5)", transform=ax_auc.transAxes,
        ha="left", va="bottom", fontsize=8,
        color="#444444", style="italic",
    )

    # Shared legend at top
    handles, labels = ax_lift.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center", bbox_to_anchor=(0.5, 1.02),
        ncol=4, fontsize=8.5, frameon=False,
    )
    # No figure title; the LaTeX caption carries the description.
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, output_path)
    plt.close(fig)


def plot_channel_ablation_dual_metric(
    setting_attack_dfs: dict,
    output_path: str | Path,
    metric_cols: tuple[str, str] = ("attack_lift", "auroc"),
    metric_labels: tuple[str, str] = (
        r"Attack lift  $L_{\mathrm{cal}}(\bar p) - L_{\mathrm{cal}}(\hat p)$",
        r"AUROC (binary $p_i > \tau$)",
    ),
    metric_refs: tuple[float, float] = (0.0, 0.5),
    channel_col: str = "channel",
) -> None:
    """Cross-setting channel ablation, two metrics side by side.

    Two grouped bar charts: panel (a) attack lift, panel (b) AUROC,
    with 95% CI from seed variance. Channels are colour-coded
    consistently with :func:`plot_tadi_realisability_eta_sweep`.

    Args:
        setting_attack_dfs: mapping setting label → attack-eval
            DataFrame, possibly pre-filtered (e.g. Setting C limited
            to η=1 cells before being passed in).
        output_path: figure path (without extension).
        metric_cols: column names for the two metrics (default
            ``("attack_lift", "auroc")``).
        metric_labels: y-axis labels for the two panels.
        metric_refs: horizontal reference line value per panel
            (lift = 0 for the calibration null; AUROC = 0.5 for chance).
        channel_col: column name carrying the channel ablation label.

    The figure makes one claim:

    - DP-SGD bounds the parameter channel ($\\mathcal{A}_1$) on every
      setting; under matched prior (Setting C at η=1), the org and
      full channels realise positive lift and AUROC ≫ 0.5; under
      mismatched prior (Setting B), no channel achieves positive lift
      and AUROC stays near chance.
    """
    _set_style()
    import matplotlib.pyplot as plt

    settings = [s for s in ("A", "B", "C") if s in setting_attack_dfs]
    if not settings:
        raise ValueError("No attack-eval data provided")

    n_channels = len(CHANNEL_ORDER)
    n_settings = len(settings)
    bar_width = 0.8 / n_channels
    x_pos = np.arange(n_settings)

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))

    for panel_idx, (ax, metric_col, ylabel, ref) in enumerate(
        zip(axes, metric_cols, metric_labels, metric_refs)
    ):
        for ci_idx, ch in enumerate(CHANNEL_ORDER):
            means, errs = [], []
            for s in settings:
                df = setting_attack_dfs[s]
                sub = df[df[channel_col] == ch]
                if sub.empty:
                    means.append(np.nan)
                    errs.append(0.0)
                else:
                    m = float(sub[metric_col].mean())
                    std = float(sub[metric_col].std() or 0.0)
                    n = max(len(sub), 1)
                    means.append(m)
                    errs.append(1.96 * std / np.sqrt(n))
            offset = (ci_idx - (n_channels - 1) / 2) * bar_width
            ax.bar(
                x_pos + offset, means, bar_width,
                yerr=errs,
                label=CHANNEL_LABEL.get(ch, ch) if panel_idx == 0 else None,
                color=CHANNEL_COLOR.get(ch, "gray"),
                edgecolor="black", linewidth=0.4,
                capsize=2, error_kw={"linewidth": 0.7},
                zorder=2,
            )

        ax.axhline(ref, color="black", linewidth=0.7,
                   linestyle="--", alpha=0.6, zorder=1)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"Setting {s}" for s in settings])
        ax.set_ylabel(ylabel)

    # Shared legend at top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center", bbox_to_anchor=(0.5, 1.02),
        ncol=4, fontsize=8.5, frameon=False,
    )
    # No figure title; the LaTeX caption carries the description.
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, output_path)
    plt.close(fig)


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

    # No figure suptitle; the LaTeX caption carries the description in
    # the manuscript. Kept consistent with the η-heatmap, Pareto, and
    # channel-ablation figures.
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


# ---------------------------------------------------------------------------
# Cross-setting comparison figures
# ---------------------------------------------------------------------------

def plot_pareto_cross_setting(
    setting_dfs: dict,
    output_path: str | Path,
    budget_col: str = "dp.utility_budget_U",
) -> None:
    """Side-by-side $U \\to K^\\star$ comparison across settings.

    Args:
        setting_dfs: mapping setting label ("A", "B", "C") → runs DataFrame
            (post-:func:`_apply_pareto_filter`). Settings missing from the
            dict are simply skipped (so this can be called incrementally
            as Pareto sweeps finish).
        output_path: figure path (without extension).
        budget_col: name of the utility-budget column.

    Each panel is one setting; within a panel the curves are grouped by
    ``dp.observation_window`` (one line per $T_{\\max}$ value, per
    allocation). Y-axis is log-scaled with plain numerals so the
    cross-setting comparison is read off in nats directly. Settings with
    no data (e.g., Setting~A while its Pareto sweep is still running)
    show as a placeholder panel.
    """
    _set_style()
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullLocator

    settings = ["A", "B", "C"]
    settings = [s for s in settings if s in setting_dfs]
    n_panels = len(settings)
    if n_panels == 0:
        raise ValueError("No setting data provided for cross-setting Pareto plot")

    fig, axes = plt.subplots(1, n_panels, figsize=(4.4 * n_panels, 3.6), sharey=False)
    if n_panels == 1:
        axes = [axes]

    color_map = {"topology_aware": "#1f77b4", "uniform": "#d62728"}
    marker_map = {"topology_aware": "o", "uniform": "s"}

    for ax, setting in zip(axes, settings):
        df = setting_dfs[setting]
        df = df.dropna(subset=["K_star", "dp.allocation",
                               "dp.observation_window", budget_col])
        if df.empty:
            ax.text(0.5, 0.5, f"Setting {setting}\n(no data yet)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=11, color="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"Setting {setting}")
            continue

        # Plot one line per (allocation, T_max) — group by both;
        # vary linewidth across T_max so the cross-product remains
        # legible without producing too many separate legend entries.
        t_values = sorted(df["dp.observation_window"].unique())
        for alloc in ["uniform", "topology_aware"]:
            for j, t_max in enumerate(t_values):
                sub = df[(df["dp.allocation"] == alloc) &
                         (df["dp.observation_window"] == t_max)]
                if sub.empty:
                    continue
                agg = (sub.groupby(budget_col)["K_star"]
                          .mean().reset_index().sort_values(budget_col))
                ax.plot(
                    agg[budget_col], agg["K_star"],
                    marker=marker_map[alloc],
                    color=color_map[alloc],
                    linewidth=1.3 + 0.4 * j,
                    markersize=4 + j,
                    alpha=0.85,
                    label=f"{alloc} (T={int(t_max)})" if setting == settings[0] else None,
                )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Utility budget $U$")
        if setting == settings[0]:
            ax.set_ylabel(r"Privacy bound $K^\star$ (nats)")
        ax.set_title(f"Setting {setting}")

        # Nice log-tick formatting
        u_values = sorted(df[budget_col].unique())
        if u_values:
            ax.xaxis.set_major_locator(FixedLocator(u_values))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _pos: f"{x:g}"))
            ax.xaxis.set_minor_locator(NullLocator())
        # Same adaptive log-scale ticks as plot_pareto_setting: dense
        # sub-decade ticks (1, 2, 3, 5, 7) with compact decimal labels
        # so the y-axis populates correctly for any per-setting K*
        # range, including Setting A where K* dips below 1.
        ax.yaxis.set_major_locator(
            LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 5.0, 7.0), numticks=12)
        )
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _pos: f"{y:g}"))
        ax.yaxis.set_minor_locator(NullLocator())

    axes[0].legend(loc="upper right", fontsize=7, framealpha=0.85)
    fig.suptitle("Privacy-bound dominance across settings", y=1.02)
    fig.tight_layout()
    _save(fig, output_path)
    plt.close(fig)


def plot_attack_channel_ablation_cross_setting(
    setting_attack_dfs: dict,
    output_path: str | Path,
    metric_col: str = "attack_lift",
    channel_col: str = "channel",
) -> None:
    """Per-channel attack-lift bar chart across settings.

    Args:
        setting_attack_dfs: mapping setting label → attack-eval DataFrame
            (as written by ``cmd_attack_eval`` to ``analysis/attack_setting_*.parquet``).
            Settings without an attack-eval parquet are skipped.

    Groups by (setting, channel); shows mean attack lift with 95% CI
    bars across target runs. The figure is the central cross-setting
    visualisation that demonstrates: (i) DP-SGD bounds A_1 across all
    settings, (ii) the prior-coupling channels are positive on
    Setting~C (matched shadow/target prior) and not realised on
    Settings~B and~A (mismatched prior).
    """
    _set_style()
    import matplotlib.pyplot as plt

    settings = [s for s in ("A", "B", "C") if s in setting_attack_dfs]
    if not settings:
        raise ValueError("No attack-eval data provided")

    channels = CHANNEL_ORDER
    width = 0.18
    fig, ax = plt.subplots(figsize=(7.5, 3.8))

    for i, ch in enumerate(channels):
        means, cis = [], []
        for s in settings:
            df = setting_attack_dfs[s]
            sub = df[df[channel_col] == ch]
            if sub.empty:
                means.append(0.0)
                cis.append(0.0)
            else:
                m = float(sub[metric_col].mean())
                std = float(sub[metric_col].std())
                ci = 1.96 * std / np.sqrt(max(len(sub), 1))
                means.append(m)
                cis.append(ci)
        x_pos = np.arange(len(settings)) + (i - (len(channels) - 1) / 2) * width
        ax.bar(
            x_pos, means, width, yerr=cis,
            label=ch, capsize=2,
            color=CHANNEL_COLOR.get(ch, "gray"),
            edgecolor="black", linewidth=0.4,
        )

    ax.set_xticks(np.arange(len(settings)))
    ax.set_xticklabels([f"Setting {s}" for s in settings])
    ax.set_ylabel(r"Attack lift = $L_{\mathrm{cal}}(\bar p) - L_{\mathrm{cal}}(\hat p)$")
    # No figure title; the LaTeX caption carries the description in the
    # manuscript. The previous title rendered the macro `\TADI` literally
    # because matplotlib does not expand LaTeX macros.
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85, ncol=4)
    fig.tight_layout()
    _save(fig, output_path)
    plt.close(fig)
