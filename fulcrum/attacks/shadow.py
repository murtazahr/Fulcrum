"""Shadow training data management for TADI.

The TADI adversary trains its regressor offline on **shadow runs** — simulated
federations on public proxy data with **known ground-truth $p_i$**. At deployment
time the trained regressor is applied to the target run's features.

This module is pure data plumbing: it does not run experiments itself. It
provides:

- :class:`ShadowSpec` — the configuration for a shadow run grid
  (which topologies, partitionings, η values, etc. to sweep).
- :func:`shadow_grid` — expand a spec into individual run configurations.
- :func:`load_shadow_dataset` — load completed shadow runs from the SQLite
  experiment database into ``(X_train, p_train, run_ids)`` arrays for
  :class:`~fulcrum.attacks.tadi.TADI` to fit on.

Shadow runs are first-class experiments (just regular runs with
``mode: shadow`` flagged in the config). The experiment runner sees them
as ordinary FL trainings; the only difference is that we record their
``p_i`` ground truth alongside the features for later regressor training.

Design ref: Redesign/03_attack_design.md §3.4
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ShadowSpec:
    """Specification for a shadow-run grid.

    Sweeps a small Cartesian product. The expansion is intentionally finite —
    shadow training quantity is itself a robustness-ablation axis (Stage 5
    plan: 100/500/1000/5000 runs).

    Attributes:
        setting: Setting name ("A", "B", or "C") — selects the dataset/adapter.
        topologies: list of topology types to sweep.
        n_clients_options: list of client counts to sweep.
        eta_values: η coupling values (Setting C only; ignored for A/B).
        alpha_values: Dirichlet α values (Setting C only).
        seeds: random seeds — one shadow run per seed.
        rounds: per-run training rounds (typically smaller than target's T_max).
        dp_levels: per-shadow noise levels — usually no DP for shadows so the
            features are clean and the regressor sees the leakage signal.
        extra: free-form metadata stored alongside each run config.
    """

    setting: str
    topologies: list[str] = field(default_factory=list)
    n_clients_options: list[int] = field(default_factory=list)
    eta_values: list[float] = field(default_factory=list)
    alpha_values: list[float] = field(default_factory=list)
    seeds: list[int] = field(default_factory=list)
    rounds: int = 50
    dp_levels: list[float | None] = field(default_factory=lambda: [None])
    extra: dict[str, Any] = field(default_factory=dict)


def shadow_grid(spec: ShadowSpec) -> list[dict[str, Any]]:
    """Expand a :class:`ShadowSpec` into individual run-config dicts.

    Each returned dict is JSON-serializable and ready to be ingested by the
    experiment runner. The runner is responsible for executing each run and
    storing per-client features + ground-truth $p_i$ in the experiments DB.

    Setting-specific axes:
    - Setting A/B: only topologies × n_clients_options × seeds × dp_levels matter.
      η/α are ignored.
    - Setting C: full sweep including η × α.
    """
    if spec.setting == "C":
        eta_axis = spec.eta_values or [0.5]
        alpha_axis = spec.alpha_values or [0.5]
    else:
        eta_axis = [None]
        alpha_axis = [None]

    n_clients_axis = spec.n_clients_options or [None]
    topo_axis = spec.topologies or ["star"]
    seed_axis = spec.seeds or [0]
    dp_axis = spec.dp_levels or [None]

    runs = []
    for topo, n_clients, eta, alpha, seed, dp in product(
        topo_axis, n_clients_axis, eta_axis, alpha_axis, seed_axis, dp_axis
    ):
        cfg: dict[str, Any] = {
            "mode": "shadow",
            "setting": spec.setting,
            "topology": topo,
            "rounds": spec.rounds,
            "seed": seed,
            "dp_noise_multiplier": dp,
            **spec.extra,
        }
        if n_clients is not None:
            cfg["n_clients"] = n_clients
        if eta is not None:
            cfg["eta"] = eta
        if alpha is not None:
            cfg["alpha"] = alpha
        runs.append(cfg)
    return runs


# ---------------------------------------------------------------------------
# Loading completed shadow runs from the experiments database
# ---------------------------------------------------------------------------

def load_shadow_dataset(
    db_path: str | Path,
    setting: str,
    runs_root: str | Path,
    channel: str = "A2_full",
    feature_set: str = "full",
    limit: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load all completed shadow runs for a setting into a regressor-ready dataset.

    Args:
        db_path: path to ``experiments.db``.
        setting: filter to ``runs.setting = setting`` AND ``runs.mode = 'shadow'``
            AND ``runs.status = 'done'``.
        runs_root: directory containing per-run artifact folders (``runs/<run_id>/``).
        channel: TADI channel ablation (passed to feature builder).
        feature_set: TADI feature-set selector (currently a no-op pass-through;
            reserved for raw vs aggregate-only ablations).
        limit: cap on number of runs to load (None = load all).

    Returns:
        ``(X, p, run_ids)`` where ``X`` is shape ``(N_total_clients, F)``
        stacking all clients across all loaded shadow runs, ``p`` is the
        corresponding ground-truth vector of length ``N_total_clients``, and
        ``run_ids`` is the per-row run identifier (useful for grouped CV).

    Notes:
        Each shadow run contributes one (X, p) block per client. We do not
        aggregate across clients within a run — each client's features are
        an independent training example for the regressor.
    """
    from fulcrum.attacks.features import build_attack_features

    db_path = Path(db_path)
    runs_root = Path(runs_root)

    if not db_path.exists():
        raise FileNotFoundError(f"experiments.db not found at {db_path}")

    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        query = (
            "SELECT run_id FROM runs "
            "WHERE setting = ? AND mode = 'shadow' AND status = 'done' "
            "ORDER BY run_id"
        )
        cur.execute(query, (setting,))
        run_ids = [row[0] for row in cur.fetchall()]
    finally:
        conn.close()

    if limit is not None:
        run_ids = run_ids[:limit]

    # First pass: discover the common per-round feature width across all
    # shadow runs. We truncate to the minimum so v1 (layer-norm-only)
    # and v2 (with class-aware blocks) shadow runs can coexist if needed.
    raw_widths: list[int] = []
    for run_id in run_ids:
        feat_path = runs_root / run_id / "features.npz"
        if not feat_path.exists():
            continue
        npz = np.load(feat_path, allow_pickle=False)
        raw_widths.append(int(npz["raw"].shape[-1]))
    if not raw_widths:
        return (np.zeros((0, 0), dtype=np.float32),
                np.zeros((0,), dtype=np.float64), [])
    common_raw_width = min(raw_widths)

    X_blocks: list[np.ndarray] = []
    p_blocks: list[np.ndarray] = []
    rid_blocks: list[str] = []

    for run_id in run_ids:
        run_dir = runs_root / run_id
        feat_path = run_dir / "features.npz"
        cfg_path = run_dir / "config.json"
        if not feat_path.exists() or not cfg_path.exists():
            continue
        npz = np.load(feat_path, allow_pickle=False)
        raw = npz["raw"]
        p_true = npz["p_true"]
        neighbors = json.loads(str(npz["neighbors_json"]))
        omega = npz["omega"] if "omega" in npz.files else None
        if omega is not None and omega.dtype != np.int64:
            omega = omega.astype(np.int64)

        try:
            X = build_attack_features(
                raw, neighbors=neighbors, omega=omega, channel=channel,
                max_round_features=common_raw_width,
            )
        except ValueError:
            continue
        X_blocks.append(X)
        p_blocks.append(p_true.astype(np.float64))
        rid_blocks.extend([run_id] * X.shape[0])

    if not X_blocks:
        return (np.zeros((0, 0), dtype=np.float32),
                np.zeros((0,), dtype=np.float64), [])

    widths = {block.shape[1] for block in X_blocks}
    if len(widths) > 1:
        raise ValueError(
            f"Inconsistent feature widths across shadow runs even after "
            f"truncation to common_raw_width={common_raw_width}: {widths}. "
            "Likely cause: different topology / organisational structures "
            "across shadow runs."
        )

    return (np.concatenate(X_blocks, axis=0),
            np.concatenate(p_blocks, axis=0),
            rid_blocks,
            common_raw_width)  # so the caller can use the same truncation on target features
