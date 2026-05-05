"""Factorial experiment manager.

Reads a sweep specification (a YAML "factorial config") that lists axes and
their values. Expands the Cartesian product into individual configs. For each
generated config:

1. Compute its content-hash run_id.
2. Check the experiments DB for status:
   - ``done``  → skip (idempotent re-runs).
   - ``running`` or ``queued`` → skip (assume another worker is on it).
   - ``failed`` → re-run if ``--retry-failed`` is passed; otherwise skip.
   - missing → execute.

Sequential execution (single GPU). For multi-GPU parallelism a separate sharded
manager would be needed; with the L40S budget this is not necessary.

Sweep YAML format:

.. code-block:: yaml

    base: configs/setting_c_canonical.yaml   # template config
    output_dir: runs
    db_path: experiments.db

    sweep:
      data.eta: [0.0, 0.25, 0.5, 0.75, 1.0]
      data.alpha: [0.5, 1.0]
      experiment.seed: [0, 1, 2]
      dp.allocation: [topology_aware, uniform]

The sweep dict's keys are dotted paths into the base config; values are lists
to iterate over. Cartesian product = total runs.

Usage:
    python scripts/run_factorial.py sweeps/setting_c_pareto.yaml
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml


def _deep_set(d: dict, dotted_key: str, value: Any) -> None:
    """Set ``d[a][b][c] = value`` from ``"a.b.c"``."""
    parts = dotted_key.split(".")
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def expand_sweep(sweep_path: Path) -> tuple[list[dict], dict[str, Any]]:
    """Read a sweep spec and return the list of expanded configs + meta.

    Returns:
        ``(configs, meta)`` where ``configs`` is a list of fully-resolved config
        dicts ready for :func:`fulcrum.config.load_config`-style parsing, and
        ``meta`` carries the ``output_dir`` and ``db_path``.
    """
    with sweep_path.open("r") as fh:
        spec = yaml.safe_load(fh)

    base_path = Path(spec["base"])
    if not base_path.is_absolute():
        base_path = sweep_path.parent / base_path
    with base_path.open("r") as fh:
        base = yaml.safe_load(fh)

    sweep = spec.get("sweep", {})
    if not sweep:
        return [base], {"output_dir": spec.get("output_dir", "runs"),
                        "db_path": spec.get("db_path", "experiments.db")}

    keys = list(sweep.keys())
    value_lists = [sweep[k] for k in keys]
    configs: list[dict] = []
    for combo in itertools.product(*value_lists):
        cfg = copy.deepcopy(base)
        for k, v in zip(keys, combo):
            _deep_set(cfg, k, v)
        configs.append(cfg)

    meta = {
        "output_dir": spec.get("output_dir", "runs"),
        "db_path": spec.get("db_path", "experiments.db"),
    }
    return configs, meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_path", type=Path, help="Sweep YAML")
    parser.add_argument("--retry-failed", action="store_true", help="Re-run failed experiments")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run, don't execute")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    from fulcrum.config import _from_dict, ExperimentConfig
    from fulcrum.runner import run_experiment
    from fulcrum import storage as st

    configs, meta = expand_sweep(args.sweep_path)
    db_path = meta["db_path"]
    print(f"Sweep expanded to {len(configs)} configs (db={db_path}, output={meta['output_dir']})")

    submitted = 0
    skipped = 0
    failed = 0
    for raw_cfg in configs:
        cfg = _from_dict(ExperimentConfig, raw_cfg)
        run_id = cfg.hash_id()
        status = st.get_run_status(db_path, run_id)
        if status == "done":
            skipped += 1
            continue
        if status in ("running", "queued") and not args.retry_failed:
            skipped += 1
            continue
        if status == "failed" and not args.retry_failed:
            skipped += 1
            continue
        if args.dry_run:
            print(f"  [dry] {run_id} setting={cfg.setting} {raw_cfg.get('experiment', {}).get('name', '')}")
            submitted += 1
            continue
        try:
            print(f"  [run] {run_id} setting={cfg.setting} ...", flush=True)
            result = run_experiment(cfg, db_path=db_path, runs_root=meta["output_dir"], device=args.device)
            print(f"        done. acc={result.get('test_accuracy', 'n/a'):.3f}" if result.get("test_accuracy") is not None else "        done.")
            submitted += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"        FAILED: {exc}", file=sys.stderr)

    print(f"\nDone. Submitted: {submitted}, skipped: {skipped}, failed: {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
