"""Analysis driver — reads experiments.db, generates Pareto figures + summary tables.

Usage examples:

.. code-block:: bash

    # Generate Pareto figure for Setting A
    python scripts/analyze.py pareto --setting A

    # Generate the η-sweep figure (Setting C only)
    python scripts/analyze.py eta-sweep

    # Export all completed runs as a Parquet table for manuscript writing
    python scripts/analyze.py export-table --setting C --out analysis/setting_c.parquet

The script writes outputs under ``analysis/`` (PNG + PDF + Parquet) and prints
the topology-aware-vs-uniform area-between-curves headline number.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def cmd_pareto(args) -> int:
    from fulcrum.analysis.figures import plot_pareto_setting
    from fulcrum.analysis.loader import load_runs_df
    from fulcrum.analysis.pareto import topology_aware_advantage

    df = load_runs_df(args.db_path, args.runs_root, setting=args.setting, status="done")
    if df.empty:
        print(f"No completed runs for setting {args.setting}", file=sys.stderr)
        return 1

    out_path = Path(args.out_dir) / f"pareto_setting_{args.setting.lower()}"
    plot_pareto_setting(df, args.setting, out_path)
    print(f"Pareto figure → {out_path}.{{png,pdf}}")

    summary = topology_aware_advantage(df)
    print(f"Topology-aware advantage (area between curves): {summary['area']:.6f}")
    print(f"  topology-aware Pareto points: {summary['ta_pareto_size']}")
    print(f"  uniform Pareto points:        {summary['unif_pareto_size']}")
    return 0


def cmd_eta_sweep(args) -> int:
    from fulcrum.analysis.figures import plot_eta_sweep
    from fulcrum.analysis.loader import load_runs_df

    df = load_runs_df(args.db_path, args.runs_root, setting="C", status="done")
    if df.empty:
        print("No completed Setting C runs", file=sys.stderr)
        return 1
    if "data.eta" not in df.columns:
        print("No 'data.eta' column found — did you run the η-sweep?", file=sys.stderr)
        return 1

    out_path = Path(args.out_dir) / "eta_sweep_setting_c"
    plot_eta_sweep(df, out_path)
    print(f"η-sweep figure → {out_path}.{{png,pdf}}")
    return 0


def cmd_export_table(args) -> int:
    from fulcrum.analysis.loader import load_runs_df

    df = load_runs_df(args.db_path, args.runs_root, setting=args.setting, status="done")
    if df.empty:
        print(f"No completed runs for setting {args.setting}", file=sys.stderr)
        return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Exported {len(df)} rows × {len(df.columns)} cols → {out_path}")
    return 0


def cmd_attack_eval(args) -> int:
    """Train TADI from shadow runs and evaluate against target runs (per channel)."""
    from fulcrum.analysis.attack_eval import evaluate_all_targets

    out_path = Path(args.out) if args.out else Path(args.out_dir) / f"attack_setting_{args.setting.lower()}.parquet"
    df = evaluate_all_targets(
        db_path=args.db_path,
        runs_root=args.runs_root,
        setting=args.setting,
        output_path=out_path,
        channels=tuple(args.channels),
        regressor=args.regressor,
        shadow_run_limit=args.shadow_limit,
    )
    if df.empty:
        print(
            f"No attack-evaluation results for setting {args.setting}. "
            f"Run shadow + target sweeps first.",
            file=sys.stderr,
        )
        return 1
    print(f"Wrote {len(df)} rows × {len(df.columns)} cols → {out_path}")
    summary = df.groupby("channel").agg({
        "calibration_loss": ["mean", "std"],
        "attack_lift":      ["mean", "std"],
        "top_k_recovery":   ["mean", "std"],
        "auroc":            ["mean", "std"],
    }).round(4)
    print()
    print("Per-channel summary across target runs:")
    print(summary.to_string())
    return 0


def cmd_summary(args) -> int:
    """Print a quick text summary of all completed runs."""
    from fulcrum.analysis.loader import load_runs_df

    for setting in (args.setting,) if args.setting else ("A", "B", "C"):
        df = load_runs_df(args.db_path, args.runs_root, setting=setting, status="done")
        if df.empty:
            continue
        print(f"\nSetting {setting}: {len(df)} completed runs")
        if "dp.allocation" in df.columns and "test_accuracy" in df.columns:
            grouped = df.groupby("dp.allocation").agg({
                "test_accuracy": ["mean", "std", "count"],
                "K_star": ["mean", "std"],
            })
            print(grouped.to_string())
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", default="experiments.db")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--out-dir", default="analysis")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_pareto = sub.add_parser("pareto", help="Per-setting Pareto figure")
    p_pareto.add_argument("--setting", required=True, choices=["A", "B", "C"])
    p_pareto.set_defaults(func=cmd_pareto)

    p_eta = sub.add_parser("eta-sweep", help="η-sweep figure for Setting C")
    p_eta.set_defaults(func=cmd_eta_sweep)

    p_export = sub.add_parser("export-table", help="Export runs DataFrame as Parquet")
    p_export.add_argument("--setting", required=True, choices=["A", "B", "C"])
    p_export.add_argument("--out", required=True)
    p_export.set_defaults(func=cmd_export_table)

    p_attack = sub.add_parser("attack-eval", help="Train TADI from shadow runs + evaluate target runs")
    p_attack.add_argument("--setting", required=True, choices=["A", "B", "C"])
    p_attack.add_argument("--regressor", default="lightgbm", choices=["lightgbm", "mlp", "linear"])
    p_attack.add_argument("--channels", nargs="+", default=["A1", "A2_topo", "A2_org", "A2_full"],
                          choices=["A1", "A2_topo", "A2_org", "A2_full"])
    p_attack.add_argument("--shadow-limit", type=int, default=None)
    p_attack.add_argument("--out", default=None)
    p_attack.set_defaults(func=cmd_attack_eval)

    p_summary = sub.add_parser("summary", help="Quick text summary of completed runs")
    p_summary.add_argument("--setting", default=None, choices=["A", "B", "C"])
    p_summary.set_defaults(func=cmd_summary)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
