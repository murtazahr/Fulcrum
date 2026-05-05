"""Run TADI training + target-run evaluation for one setting.

Trains one TADI regressor per channel ablation from completed shadow runs,
applies each to every completed target run, and writes the per-(target,
channel) attack metrics to a Parquet file.

Run AFTER both shadow and target sweeps have completed for the setting.

Usage:
    python scripts/run_attack_eval.py --setting C
    python scripts/run_attack_eval.py --setting A --regressor lightgbm --out analysis/attack_a.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting", required=True, choices=["A", "B", "C"])
    parser.add_argument("--db-path", default="experiments.db")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument(
        "--regressor",
        default="lightgbm",
        choices=["lightgbm", "mlp", "linear"],
        help="TADI regressor backend (default: lightgbm)",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        default=["A1", "A2_topo", "A2_org", "A2_full"],
        choices=["A1", "A2_topo", "A2_org", "A2_full"],
        help="Channel ablations to evaluate (default: all four)",
    )
    parser.add_argument(
        "--shadow-limit",
        type=int,
        default=None,
        help="Cap number of shadow runs to load for training (default: all)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output Parquet path (default: analysis/attack_setting_{X}.parquet)",
    )
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else Path("analysis") / f"attack_setting_{args.setting.lower()}.parquet"

    from fulcrum.analysis.attack_eval import evaluate_all_targets

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
            f"No attack-evaluation results for setting {args.setting}.\n"
            f"  Possible reasons:\n"
            f"   - No completed shadow runs (check `fulcrum status --setting {args.setting} --mode shadow`)\n"
            f"   - No completed target runs (check `fulcrum status --setting {args.setting} --mode target`)\n"
            f"   - Feature-width mismatch between shadow and target (different n_clients?)",
            file=sys.stderr,
        )
        return 1

    print(f"Wrote {len(df)} rows × {len(df.columns)} cols → {out_path}")
    print()
    summary = df.groupby("channel").agg({
        "calibration_loss": ["mean", "std"],
        "attack_lift":      ["mean", "std"],
        "top_k_recovery":   ["mean", "std"],
        "auroc":            ["mean", "std"],
    }).round(4)
    print("Per-channel summary across target runs:")
    print(summary.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
