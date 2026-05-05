"""Run a single experiment from a YAML config.

Thin wrapper around :func:`fulcrum.runner.run_experiment` for shell use.

Usage:
    python scripts/run_experiment.py configs/setting_a_canonical.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_path", type=Path, help="YAML config")
    parser.add_argument("--db-path", default="experiments.db", help="SQLite DB path")
    parser.add_argument("--device", default=None, help="cuda or cpu (auto-detect by default)")
    args = parser.parse_args()

    from fulcrum.config import load_config
    from fulcrum.runner import run_experiment

    cfg = load_config(args.config_path)
    print(f"Running config={args.config_path.name} setting={cfg.setting} mode={cfg.experiment.mode}")
    result = run_experiment(cfg, db_path=args.db_path, device=args.device)
    print(json.dumps(
        {k: v for k, v in result.items() if not isinstance(v, list)},
        indent=2, default=str,
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
