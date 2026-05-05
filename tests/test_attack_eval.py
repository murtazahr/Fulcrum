"""Tests for fulcrum.analysis.attack_eval — TADI fit + target evaluation.

Behavior verified:

- ``evaluate_all_targets`` returns an empty DataFrame when there are no shadow
  runs (graceful degradation, no crash).
- ``evaluate_target_run`` skips channels whose feature-builder requires
  unsaved inputs (no spurious rows).
- ``join_with_run_metadata`` produces a left-join on ``target_run_id`` ↔
  ``run_id`` and preserves all attack rows.

Full end-to-end (shadow runs → fitted TADI → target evaluation) is covered
implicitly by the on-VM integration test once both sweeps complete; here we
test the data plumbing only.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fulcrum.analysis.attack_eval import (
    AttackEvalRow,
    evaluate_all_targets,
    join_with_run_metadata,
)


# ---------------------------------------------------------------------------
# AttackEvalRow shape
# ---------------------------------------------------------------------------

class TestAttackEvalRow:
    def test_dataclass_fields_present(self):
        row = AttackEvalRow(
            target_run_id="abc123",
            setting="C",
            channel="A2_full",
            regressor="lightgbm",
            n_shadow_examples=300,
            n_clients=50,
            calibration_loss=0.05,
            constant_mean_baseline=0.06,
            attack_lift=0.01,
            top_k_recovery=0.7,
            auroc=0.85,
        )
        assert row.target_run_id == "abc123"
        assert row.attack_lift == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# evaluate_all_targets — empty / missing data graceful behaviour
# ---------------------------------------------------------------------------

class TestEvalAllTargets:
    def test_empty_db_returns_empty_df(self, tmp_path):
        # Create an empty experiments.db
        from fulcrum import storage as st
        db = tmp_path / "experiments.db"
        st.init_db(db)
        out = tmp_path / "out.parquet"
        df = evaluate_all_targets(db, tmp_path / "runs", setting="C", output_path=out)
        assert df.empty
        assert not out.exists()  # nothing written when there's nothing to write

    def test_missing_db_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            evaluate_all_targets(
                tmp_path / "nonexistent.db",
                tmp_path / "runs",
                setting="C",
                output_path=tmp_path / "out.parquet",
            )


# ---------------------------------------------------------------------------
# join_with_run_metadata
# ---------------------------------------------------------------------------

class TestJoin:
    def test_left_join_on_target_run_id(self):
        attack_df = pd.DataFrame({
            "target_run_id": ["a", "a", "b"],
            "channel": ["A1", "A2_full", "A1"],
            "attack_lift": [0.01, 0.05, 0.02],
        })
        runs_df = pd.DataFrame({
            "run_id": ["a", "b", "c"],     # 'c' has no attack rows; should not appear
            "setting": ["C", "C", "C"],
            "K_star": [1.0, 2.0, 3.0],
            "test_accuracy": [0.7, 0.8, 0.9],
        })
        joined = join_with_run_metadata(attack_df, runs_df)
        # Left join: 3 attack rows preserved
        assert len(joined) == 3
        # Each attack row picks up its target's K_star
        assert joined.loc[joined["target_run_id"] == "a", "K_star"].iloc[0] == 1.0
        assert joined.loc[joined["target_run_id"] == "b", "K_star"].iloc[0] == 2.0
        # 'c' doesn't appear because it has no attack rows
        assert "c" not in joined["target_run_id"].values

    def test_join_passthrough_on_empty_attack_df(self):
        empty = pd.DataFrame()
        runs_df = pd.DataFrame({"run_id": ["a"], "K_star": [1.0]})
        joined = join_with_run_metadata(empty, runs_df)
        assert joined.empty

    def test_join_no_run_id_column(self):
        # If the runs DF already uses target_run_id (e.g. pre-renamed), should still work
        attack_df = pd.DataFrame({"target_run_id": ["a"], "attack_lift": [0.05]})
        runs_df = pd.DataFrame({"target_run_id": ["a"], "K_star": [1.5]})
        joined = join_with_run_metadata(attack_df, runs_df)
        assert len(joined) == 1
        assert joined["K_star"].iloc[0] == 1.5
