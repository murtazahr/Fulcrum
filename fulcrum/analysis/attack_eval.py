"""Shadow → TADI training + target-run evaluation pipeline.

The structural gap that this module fills:

- Target runs collect per-round per-client features and persist them in
  ``runs/<run_id>/features.npz`` (alongside ground-truth $p_i$ for evaluation).
- Shadow runs (``mode: shadow`` in their config) do the same — but the adversary
  is allowed to use the shadow runs' $p_i$ as training labels because the shadow
  data is public-by-assumption (see Stage 3 design §3.4).
- This module trains one TADI regressor per channel ($\\mathcal{A}_1$,
  $\\mathcal{A}_2^{\\text{topo}}$, $\\mathcal{A}_2^{\\text{org}}$,
  $\\mathcal{A}_2^{\\text{full}}$) on the shadow data, then applies each
  trained regressor to every target run and computes the attack-metric panel
  (calibration loss, attack lift, top-k recovery, AUROC).

Output: a long-format Parquet table with one row per (target_run_id, channel)
pair, ready to be joined with the runs DataFrame for plotting.

Design ref: docs/attack_design.md §3.4–3.7
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fulcrum.analysis.loader import load_run_artifacts
from fulcrum.attacks.features import build_attack_features
from fulcrum.attacks.shadow import load_shadow_dataset
from fulcrum.attacks.tadi import TADI
from fulcrum import storage as st


# ---------------------------------------------------------------------------
# Result row schema
# ---------------------------------------------------------------------------

@dataclass
class AttackEvalRow:
    """One (target_run, channel, regressor) attack evaluation result."""

    target_run_id: str
    setting: str
    channel: str
    regressor: str
    n_shadow_examples: int
    n_clients: int
    calibration_loss: float
    constant_mean_baseline: float
    attack_lift: float
    top_k_recovery: float
    auroc: float


# ---------------------------------------------------------------------------
# Train TADI per channel
# ---------------------------------------------------------------------------

def fit_tadi_per_channel(
    db_path: str | Path,
    runs_root: str | Path,
    setting: str,
    channels: tuple[str, ...] = ("A1", "A2_topo", "A2_org", "A2_full"),
    regressor: str = "lightgbm",
    shadow_run_limit: int | None = None,
) -> dict[str, TADI]:
    """Train one :class:`TADI` regressor per channel from completed shadow runs.

    Args:
        db_path: experiments.db path.
        runs_root: per-run artifact directory.
        setting: ``"A"``, ``"B"``, or ``"C"``.
        channels: which channel ablations to fit (omit any to skip).
        regressor: ``"lightgbm"``, ``"mlp"``, or ``"linear"``.
        shadow_run_limit: cap on shadow runs to load (None = all).

    Returns:
        Dict mapping channel name → fitted :class:`TADI`. Channels with no
        successfully-loaded shadow data are silently omitted.
    """
    fitted: dict[str, TADI] = {}
    for channel in channels:
        X, p, run_ids = load_shadow_dataset(
            db_path=db_path,
            runs_root=runs_root,
            setting=setting,
            channel=channel,
            limit=shadow_run_limit,
        )
        if X.shape[0] == 0:
            continue
        attack = TADI(regressor=regressor)
        attack.fit(X, p)
        # Stash provenance on the object for the eval row
        attack.n_shadow_examples = int(X.shape[0])  # type: ignore[attr-defined]
        attack.n_shadow_runs = len(set(run_ids))    # type: ignore[attr-defined]
        attack.feature_width = int(X.shape[1])      # type: ignore[attr-defined]
        fitted[channel] = attack
    return fitted


# ---------------------------------------------------------------------------
# Apply trained TADIs to one target run
# ---------------------------------------------------------------------------

def evaluate_target_run(
    target_run_id: str,
    runs_root: str | Path,
    fitted_tadis: dict[str, TADI],
    setting: str,
) -> list[AttackEvalRow]:
    """Run all fitted TADIs against one target run; return per-channel rows.

    Skips channels whose feature builder requires inputs the target run didn't
    save (e.g., ``A2_topo`` against a setting without topology metadata) or
    whose feature width disagrees with the trained regressor (different
    n_clients between shadow and target).
    """
    artifacts = load_run_artifacts(target_run_id, runs_root)
    if "raw_features" not in artifacts or "p_true" not in artifacts:
        return []
    raw = artifacts["raw_features"]
    p_true = artifacts["p_true"]
    neighbors = artifacts.get("neighbors")
    omega = artifacts.get("omega")

    rows: list[AttackEvalRow] = []
    for channel, attack in fitted_tadis.items():
        try:
            X = build_attack_features(raw, neighbors=neighbors, omega=omega, channel=channel)
        except ValueError:
            # Channel needs inputs this run didn't persist
            continue
        expected_width = getattr(attack, "feature_width", None)
        if expected_width is not None and X.shape[1] != expected_width:
            # Width mismatch (different n_clients / model / org structure between shadow and target)
            continue
        p_hat = attack.predict(X)
        metrics = TADI.metrics(p_hat, p_true)
        rows.append(AttackEvalRow(
            target_run_id=target_run_id,
            setting=setting,
            channel=channel,
            regressor=attack.regressor,
            n_shadow_examples=int(getattr(attack, "n_shadow_examples", 0)),
            n_clients=int(metrics["n_clients"]),
            calibration_loss=float(metrics["calibration_loss"]),
            constant_mean_baseline=float(metrics["constant_mean_baseline"]),
            attack_lift=float(metrics["attack_lift"]),
            top_k_recovery=float(metrics["top_k_recovery"]),
            auroc=float(metrics["auroc"]),
        ))
    return rows


# ---------------------------------------------------------------------------
# End-to-end: fit + evaluate all targets for a setting
# ---------------------------------------------------------------------------

def evaluate_all_targets(
    db_path: str | Path,
    runs_root: str | Path,
    setting: str,
    output_path: str | Path,
    channels: tuple[str, ...] = ("A1", "A2_topo", "A2_org", "A2_full"),
    regressor: str = "lightgbm",
    shadow_run_limit: int | None = None,
) -> pd.DataFrame:
    """Fit TADI per channel from shadow runs, apply to all target runs, save Parquet.

    Args:
        db_path: experiments.db.
        runs_root: per-run artifact directory.
        setting: which setting to evaluate (filters both shadow and target runs).
        output_path: Parquet file path for the results.
        channels: which channel ablations to evaluate.
        regressor: which regressor backend to use.
        shadow_run_limit: cap on shadow runs (None = all).

    Returns:
        DataFrame with one row per (target_run, channel). Empty if no shadow
        runs are available for this setting yet.

    Raises:
        FileNotFoundError: if ``db_path`` does not exist.
    """
    db_path = Path(db_path)
    runs_root = Path(runs_root)
    output_path = Path(output_path)

    if not db_path.exists():
        raise FileNotFoundError(f"experiments.db not found at {db_path}")

    fitted = fit_tadi_per_channel(
        db_path=db_path, runs_root=runs_root, setting=setting,
        channels=channels, regressor=regressor, shadow_run_limit=shadow_run_limit,
    )
    if not fitted:
        return pd.DataFrame()

    target_runs = st.list_runs(db_path, setting=setting, mode="target", status="done")
    rows: list[dict[str, Any]] = []
    for run in target_runs:
        for r in evaluate_target_run(run["run_id"], runs_root, fitted, setting):
            rows.append(asdict(r))

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame.from_records(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return df


def join_with_run_metadata(
    attack_df: pd.DataFrame,
    runs_df: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join the attack-evaluation rows with the runs DataFrame.

    The result has every (target_run, channel) row enriched with the target
    run's config + summary metrics (allocation, U, T_max, K_star, etc.) — the
    shape needed for joint privacy/utility/attack figures.
    """
    if attack_df.empty or runs_df.empty:
        return attack_df
    if "run_id" in runs_df.columns:
        right = runs_df.rename(columns={"run_id": "target_run_id"})
    else:
        right = runs_df.copy()
    return attack_df.merge(right, on="target_run_id", how="left")
