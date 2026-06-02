"""Local figure regeneration driver — reads the parquets in analysis/ and
calls the figure functions directly, without touching the experiments.db
that lives on the VM.

Usage (from project root):
    python scripts/regen_figures_local.py

Produces, under analysis/ (and copies into paper/figures/):
    eta_gap_heatmap_setting_c.pdf
    pareto_setting_a.pdf
    pareto_setting_b.pdf
    pareto_setting_c.pdf
    channel_ablation_cross_setting.pdf
    attack_lift_vs_K_setting_c.pdf
    setting_a_util.{csv,md,tex}
    setting_b_util.{csv,md,tex}
    setting_c_util.{csv,md,tex}

The filters mirror PARETO_FILTER / ETA_SWEEP_FILTER from scripts/analyze.py.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from fulcrum.analysis.figures import (
    plot_channel_ablation_dual_metric,
    plot_eta_gap_heatmap,
    plot_pareto_setting,
    plot_tadi_realisability_trajectory,
    utility_consistency_latex,
    utility_consistency_table,
)


REPO = Path(__file__).resolve().parent.parent
ANALYSIS = REPO / "analysis"
FIGURES = REPO / "paper" / "figures"

PARETO_FILTER = {
    "A": {"dp.observation_window": [25, 50, 100]},
    "B": {"dp.observation_window": [25, 50, 100]},
    "C": {
        "topology.type": "hierarchical",
        "data.eta": 0.5,
        "dp.utility_budget_U": [0.05, 0.1, 0.2, 0.4, 0.8, 1.6],
        "dp.observation_window": [25, 50, 100],
    },
}

ETA_SWEEP_FILTER = {
    "dp.utility_budget_U": 0.5,
    "dp.observation_window": 100,
}


def _apply(df: pd.DataFrame, flt: dict) -> pd.DataFrame:
    for col, val in flt.items():
        if col not in df.columns:
            continue
        if isinstance(val, (list, tuple, set)):
            df = df[df[col].isin(list(val))]
        else:
            df = df[df[col] == val]
    return df


def regen_pareto_and_util():
    for setting in ("A", "B", "C"):
        df = pd.read_parquet(ANALYSIS / f"setting_{setting.lower()}_all.parquet")
        df = _apply(df, PARETO_FILTER[setting])
        if df.empty:
            print(f"  Setting {setting}: NO DATA AFTER FILTER, skipping")
            continue
        # Pareto figure
        out = ANALYSIS / f"pareto_setting_{setting.lower()}"
        plot_pareto_setting(df, setting, out)
        print(f"  Pareto Setting {setting}: {out}.{{pdf,png}}")

        # Utility table — all three formats
        tbl = utility_consistency_table(df)
        (ANALYSIS / f"setting_{setting.lower()}_util.csv").write_text(
            tbl.to_csv(index=False))
        (ANALYSIS / f"setting_{setting.lower()}_util.md").write_text(
            tbl.to_markdown(index=False))
        (ANALYSIS / f"setting_{setting.lower()}_util.tex").write_text(
            utility_consistency_latex(tbl, setting))
        print(f"  Utility table Setting {setting}: setting_{setting.lower()}_util.{{csv,md,tex}}")


def regen_eta_heatmap():
    df = pd.read_parquet(ANALYSIS / "setting_c_all.parquet")
    df = _apply(df, ETA_SWEEP_FILTER)
    if df.empty:
        print("  η-heatmap: NO DATA AFTER FILTER, skipping")
        return
    # Compute analytic asymptote: an/U where a = T C^2 / (2|B|^2)
    T_max, C_, B, n, U = 100, 1.0, 64, 50, 0.5
    asym = T_max * C_**2 * n / (2 * B**2 * U)
    out = ANALYSIS / "eta_gap_heatmap_setting_c"
    plot_eta_gap_heatmap(df, out, sort_by="max_gap", asymptote_anU=asym)
    print(f"  η-heatmap: {out}.{{pdf,png}}  (asymptote an/U = {asym:.4f})")


def regen_tadi_realisability():
    """Setting C η-sweep: attack lift + AUROC vs η for the four channels."""
    attack = pd.read_parquet(ANALYSIS / "attack_setting_c.parquet")
    runs = pd.read_parquet(ANALYSIS / "setting_c_all.parquet")
    joined = attack.merge(
        runs[["run_id", "data.eta"]].rename(columns={"run_id": "target_run_id"}),
        on="target_run_id", how="left",
    )
    out = ANALYSIS / "tadi_realisability_setting_c"
    plot_tadi_realisability_trajectory(joined, out)
    print(f"  TADI realisability trajectory (Setting C): {out}.{{pdf,png}}")


def regen_cross_setting_channel_ablation():
    """Cross-setting bars: Setting C restricted to η=1 cells (matched-prior
    ceiling), Setting B left as-is (mismatched-prior native partitioning)."""
    setting_attack: dict[str, pd.DataFrame] = {}

    # Setting C — restrict to η=1 cells
    if (ANALYSIS / "attack_setting_c.parquet").exists():
        attack_c = pd.read_parquet(ANALYSIS / "attack_setting_c.parquet")
        runs_c = pd.read_parquet(ANALYSIS / "setting_c_all.parquet")
        joined_c = attack_c.merge(
            runs_c[["run_id", "data.eta"]].rename(columns={"run_id": "target_run_id"}),
            on="target_run_id", how="left",
        )
        setting_attack["C"] = joined_c[joined_c["data.eta"] == 1.0]

    # Setting B — all runs (no η dependence in the native partitioning)
    if (ANALYSIS / "attack_setting_b.parquet").exists():
        setting_attack["B"] = pd.read_parquet(ANALYSIS / "attack_setting_b.parquet")

    if not setting_attack:
        print("  Cross-setting channel ablation: NO attack parquets, skipping")
        return
    out = ANALYSIS / "channel_ablation_cross_setting"
    plot_channel_ablation_dual_metric(setting_attack, out)
    print(f"  Cross-setting channel ablation (lift + AUROC): {out}.{{pdf,png}}  "
          f"(settings={sorted(setting_attack)}, C restricted to η=1)")


def copy_to_paper_figures():
    for name in (
        "eta_gap_heatmap_setting_c.pdf",
        "pareto_setting_a.pdf",
        "pareto_setting_b.pdf",
        "pareto_setting_c.pdf",
        "tadi_realisability_setting_c.pdf",
        "channel_ablation_cross_setting.pdf",
    ):
        src = ANALYSIS / name
        dst = FIGURES / name
        if src.exists():
            shutil.copyfile(src, dst)
            print(f"  copied → {dst}")


def main():
    print("== Pareto + utility tables ==")
    regen_pareto_and_util()
    print("\n== η-heatmap ==")
    regen_eta_heatmap()
    print("\n== TADI realisability (Setting C η-sweep) ==")
    regen_tadi_realisability()
    print("\n== Cross-setting channel ablation (lift + AUROC bars) ==")
    regen_cross_setting_channel_ablation()
    print("\n== Copy to paper/figures/ ==")
    copy_to_paper_figures()
    print("\nDone.")


if __name__ == "__main__":
    main()
