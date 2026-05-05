"""SQLite-backed experiment log + per-run artifact I/O.

The experiments database (``experiments.db``) is the single queryable source
of truth: every started/completed/failed run is recorded with its config hash,
status, summary metrics, and pointers to the per-run artifact directory.

Per-run artifacts live in ``runs/<run_id>/``:

- ``config.json`` — the full config dict
- ``features.npz`` — per-round per-client feature buffer + ground truth
- ``metrics.json`` — test loss, attack metrics, runtime, ε, etc.
- ``log.txt`` — stdout/stderr from the runner

This pattern was chosen over MLflow / Weights & Biases for two reasons:
queryability (SQLite is just SQL) and zero external dependencies.

Schema migration is intentionally absent — for a research codebase the schema
is small and we re-create the DB if it changes. ``init_db()`` is idempotent
(``CREATE TABLE IF NOT EXISTS``).
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id            TEXT PRIMARY KEY,
    config_hash       TEXT NOT NULL,
    setting           TEXT NOT NULL,
    mode              TEXT NOT NULL,
    status            TEXT NOT NULL DEFAULT 'queued',
    created_at        REAL NOT NULL,
    started_at        REAL,
    completed_at      REAL,
    n_clients         INTEGER,
    rounds            INTEGER,
    -- Summary metrics (NULL until completed)
    test_accuracy     REAL,
    final_loss        REAL,
    K_star            REAL,
    K_uniform         REAL,
    epsilon           REAL,
    -- Pointers
    output_dir        TEXT NOT NULL,
    config_path       TEXT,
    features_path     TEXT,
    metrics_path      TEXT,
    log_path          TEXT,
    -- Free-form
    notes             TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_setting_mode ON runs (setting, mode);
CREATE INDEX IF NOT EXISTS idx_runs_status      ON runs (status);
CREATE INDEX IF NOT EXISTS idx_runs_config_hash ON runs (config_hash);
"""


def init_db(db_path: str | Path) -> None:
    """Create the experiments DB at ``db_path`` if it does not exist."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as conn:
        conn.executescript(SCHEMA)
        conn.commit()


@contextmanager
def connect(db_path: str | Path) -> Iterator[sqlite3.Connection]:
    """Context manager yielding a SQLite connection (auto-init schema)."""
    init_db(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Run lifecycle
# ---------------------------------------------------------------------------

def register_run(
    db_path: str | Path,
    cfg,
    output_dir: str | Path,
) -> str:
    """Create a queued run row from an :class:`~fulcrum.config.ExperimentConfig`.

    Returns the ``run_id`` (= cfg.hash_id()). If a row with this ID already
    exists, returns the existing one without modification — the runner can then
    decide whether to re-execute or skip based on the current status.
    """
    run_id = cfg.hash_id()
    with connect(db_path) as conn:
        existing = conn.execute("SELECT run_id FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if existing is not None:
            return run_id
        conn.execute(
            """
            INSERT INTO runs (run_id, config_hash, setting, mode, status, created_at, output_dir, n_clients, rounds)
            VALUES (?, ?, ?, ?, 'queued', ?, ?, ?, ?)
            """,
            (
                run_id,
                run_id,
                cfg.setting,
                cfg.experiment.mode,
                time.time(),
                str(output_dir),
                cfg.data.n_clients,
                cfg.training.rounds,
            ),
        )
        conn.commit()
    return run_id


def mark_started(db_path: str | Path, run_id: str) -> None:
    with connect(db_path) as conn:
        conn.execute(
            "UPDATE runs SET status = 'running', started_at = ? WHERE run_id = ?",
            (time.time(), run_id),
        )
        conn.commit()


def mark_done(
    db_path: str | Path,
    run_id: str,
    *,
    test_accuracy: float | None = None,
    final_loss: float | None = None,
    K_star: float | None = None,
    K_uniform: float | None = None,
    epsilon: float | None = None,
    config_path: str | Path | None = None,
    features_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
    log_path: str | Path | None = None,
    notes: str | None = None,
) -> None:
    with connect(db_path) as conn:
        conn.execute(
            """
            UPDATE runs SET status = 'done', completed_at = ?,
                test_accuracy = ?, final_loss = ?, K_star = ?, K_uniform = ?, epsilon = ?,
                config_path = ?, features_path = ?, metrics_path = ?, log_path = ?,
                notes = COALESCE(?, notes)
            WHERE run_id = ?
            """,
            (
                time.time(),
                test_accuracy, final_loss, K_star, K_uniform, epsilon,
                str(config_path) if config_path else None,
                str(features_path) if features_path else None,
                str(metrics_path) if metrics_path else None,
                str(log_path) if log_path else None,
                notes,
                run_id,
            ),
        )
        conn.commit()


def mark_failed(db_path: str | Path, run_id: str, error: str) -> None:
    with connect(db_path) as conn:
        conn.execute(
            "UPDATE runs SET status = 'failed', completed_at = ?, notes = ? WHERE run_id = ?",
            (time.time(), error[:1000], run_id),
        )
        conn.commit()


def get_run_status(db_path: str | Path, run_id: str) -> str | None:
    """Return the current status of a run, or None if not registered."""
    with connect(db_path) as conn:
        row = conn.execute("SELECT status FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return row["status"] if row is not None else None


def list_runs(
    db_path: str | Path,
    setting: str | None = None,
    mode: str | None = None,
    status: str | None = None,
) -> list[dict[str, Any]]:
    """Return all runs matching the given filters as a list of dicts."""
    clauses = []
    params: list[Any] = []
    if setting:
        clauses.append("setting = ?")
        params.append(setting)
    if mode:
        clauses.append("mode = ?")
        params.append(mode)
    if status:
        clauses.append("status = ?")
        params.append(status)
    sql = "SELECT * FROM runs"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY created_at"
    with connect(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]


# ---------------------------------------------------------------------------
# Per-run artifact I/O
# ---------------------------------------------------------------------------

def write_config_json(run_dir: Path, cfg) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "config.json"
    with path.open("w") as fh:
        json.dump(asdict(cfg), fh, indent=2, default=str)
    return path


def write_metrics_json(run_dir: Path, metrics: dict[str, Any]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "metrics.json"
    with path.open("w") as fh:
        json.dump(metrics, fh, indent=2, default=_json_default)
    return path


def write_features_npz(
    run_dir: Path,
    raw_features: np.ndarray,
    p_true: np.ndarray,
    neighbors: list[list[int]],
    omega: np.ndarray | None = None,
) -> Path:
    """Save the per-round per-client feature buffer + ground truth for TADI.

    Format expected by :func:`fulcrum.attacks.shadow.load_shadow_dataset`.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "features.npz"
    payload = {
        "raw": raw_features.astype(np.float32, copy=False),
        "p_true": np.asarray(p_true, dtype=np.float64),
        "neighbors_json": np.array(json.dumps(neighbors)),
    }
    if omega is not None:
        payload["omega"] = np.asarray(omega, dtype=np.int64)
    np.savez_compressed(path, **payload)
    return path


def _json_default(obj):
    """JSON encoder fallback for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
