"""Load runs from experiments.db + per-run artifacts into pandas DataFrames.

The analysis pipeline operates on flattened DataFrames where each row is one
completed run. Config fields are flattened into dotted columns
(``dp.allocation``, ``data.eta``, etc.) for easy filtering.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Recursively flatten nested dict into dotted keys."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if prefix == "" else f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def load_runs_df(
    db_path: str | Path,
    runs_root: str | Path,
    setting: str | None = None,
    mode: str | None = None,
    status: str = "done",
) -> pd.DataFrame:
    """Load completed runs into a DataFrame with flattened config + DB metrics.

    Args:
        db_path: path to experiments.db.
        runs_root: directory containing per-run subdirectories.
        setting: optional filter to one setting.
        mode: optional filter to ``target`` or ``shadow``.
        status: status filter (default ``"done"``).

    Returns:
        DataFrame with columns from ``runs`` table (run_id, status, K_star,
        K_uniform, test_accuracy, etc.) plus flattened config columns.
    """
    db_path = Path(db_path)
    runs_root = Path(runs_root)

    clauses = ["status = ?"]
    params: list[Any] = [status]
    if setting:
        clauses.append("setting = ?")
        params.append(setting)
    if mode:
        clauses.append("mode = ?")
        params.append(mode)
    sql = f"SELECT * FROM runs WHERE {' AND '.join(clauses)} ORDER BY created_at"

    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = [dict(r) for r in conn.execute(sql, params).fetchall()]

    records: list[dict[str, Any]] = []
    for row in rows:
        run_dir = runs_root / row["run_id"]
        cfg_path = run_dir / "config.json"
        if cfg_path.exists():
            with cfg_path.open() as fh:
                cfg = json.load(fh)
            cfg_flat = _flatten(cfg)
        else:
            cfg_flat = {}
        records.append({**row, **cfg_flat})

    return pd.DataFrame.from_records(records)


def load_run_artifacts(
    run_id: str,
    runs_root: str | Path,
) -> dict[str, Any]:
    """Load a single run's per-artifact data.

    Returns a dict with keys: ``config`` (dict), ``metrics`` (dict),
    ``raw_features`` (np.ndarray), ``p_true`` (np.ndarray),
    ``neighbors`` (list of lists), ``omega`` (np.ndarray or None).
    """
    runs_root = Path(runs_root)
    run_dir = runs_root / run_id

    out: dict[str, Any] = {"run_id": run_id}
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        with cfg_path.open() as fh:
            out["config"] = json.load(fh)
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        with metrics_path.open() as fh:
            out["metrics"] = json.load(fh)
    feat_path = run_dir / "features.npz"
    if feat_path.exists():
        npz = np.load(feat_path, allow_pickle=False)
        out["raw_features"] = npz["raw"]
        out["p_true"] = npz["p_true"]
        out["neighbors"] = json.loads(str(npz["neighbors_json"]))
        if "omega" in npz.files:
            out["omega"] = npz["omega"]
    return out
