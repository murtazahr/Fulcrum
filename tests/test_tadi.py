"""Tests for TADI metrics and feature builders.

The training-side regressor backends are tested implicitly when the experiment
runner produces real shadow data; the math here (calibration loss, attack lift,
top-k recovery, AUROC, constant-mean baseline equivalences) is what we verify
analytically.
"""

from __future__ import annotations

import numpy as np
import pytest

from fulcrum.attacks.features import build_attack_features
from fulcrum.attacks.tadi import (
    TADI,
    attack_lift,
    auroc_membership,
    calibration_loss,
    constant_mean_baseline_loss,
    top_k_recovery,
)


# ---------------------------------------------------------------------------
# Calibration loss & baselines
# ---------------------------------------------------------------------------

class TestCalibrationLoss:
    def test_perfect_predictor_zero_loss(self):
        p = np.array([0.1, 0.5, 0.9])
        assert calibration_loss(p, p) == 0.0

    def test_constant_mean_equals_variance(self):
        # MSE of the mean predictor is exactly the variance of the targets.
        p = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        baseline = constant_mean_baseline_loss(p)
        assert baseline == pytest.approx(float(np.var(p)))

    def test_attack_lift_is_baseline_minus_loss(self):
        p_true = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        p_hat = p_true.copy()  # perfect predictor
        # Lift = baseline - 0 = baseline
        assert attack_lift(p_hat, p_true) == pytest.approx(constant_mean_baseline_loss(p_true))

    def test_attack_lift_zero_for_constant_predictor(self):
        # When the regressor predicts the mean, lift should be exactly zero.
        p_true = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        p_hat = np.full_like(p_true, p_true.mean())
        assert attack_lift(p_hat, p_true) == pytest.approx(0.0, abs=1e-12)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            calibration_loss(np.array([0.1, 0.2]), np.array([0.1, 0.2, 0.3]))


# ---------------------------------------------------------------------------
# Top-k recovery
# ---------------------------------------------------------------------------

class TestTopK:
    def test_perfect_ranking(self):
        p_true = np.array([0.1, 0.4, 0.5, 0.9, 0.6])
        p_hat = p_true.copy()
        assert top_k_recovery(p_hat, p_true, k=2) == 1.0

    def test_completely_wrong_ranking(self):
        p_true = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        # Reverse ranking — top-1 should miss
        p_hat = -p_true
        assert top_k_recovery(p_hat, p_true, k=1) == 0.0

    def test_k_equals_n(self):
        # Every client is in both top-n sets trivially
        p_true = np.array([0.1, 0.2, 0.3])
        p_hat = np.array([0.5, 0.4, 0.6])
        assert top_k_recovery(p_hat, p_true, k=3) == 1.0

    def test_k_out_of_range(self):
        p = np.array([0.1, 0.2, 0.3])
        with pytest.raises(ValueError):
            top_k_recovery(p, p, k=0)
        with pytest.raises(ValueError):
            top_k_recovery(p, p, k=4)


# ---------------------------------------------------------------------------
# AUROC
# ---------------------------------------------------------------------------

class TestAUROC:
    def test_perfect_separation(self):
        p_true = np.array([0.1, 0.2, 0.7, 0.8, 0.9])
        p_hat = p_true.copy()
        assert auroc_membership(p_hat, p_true, threshold=0.5) == 1.0

    def test_all_one_class_returns_chance(self):
        # If everyone is below threshold, AUROC is undefined; we return 0.5.
        p_true = np.array([0.1, 0.2, 0.3])
        p_hat = np.array([0.5, 0.4, 0.6])
        assert auroc_membership(p_hat, p_true, threshold=0.5) == 0.5

    def test_random_predictor_near_chance(self):
        # On a small dataset the random AUROC has high variance, but mean over
        # 50 trials should be close to 0.5.
        rng = np.random.default_rng(0)
        p_true = rng.uniform(size=200)
        aurocs = [auroc_membership(rng.uniform(size=200), p_true) for _ in range(50)]
        assert 0.4 <= float(np.mean(aurocs)) <= 0.6


# ---------------------------------------------------------------------------
# TADI metric panel
# ---------------------------------------------------------------------------

class TestTADIPanel:
    def test_perfect_predictor_panel(self):
        p_true = np.linspace(0.05, 0.95, 12)
        out = TADI.metrics(p_true, p_true, top_k=3)
        assert out["calibration_loss"] == 0.0
        assert out["attack_lift"] == pytest.approx(constant_mean_baseline_loss(p_true))
        assert out["top_k_recovery"] == 1.0
        assert out["n_clients"] == 12
        assert out["top_k"] == 3

    def test_default_top_k(self):
        # Default top_k is ceil(n/3); for n=9 → 3.
        out = TADI.metrics(np.zeros(9), np.linspace(0, 1, 9))
        assert out["top_k"] == 3


# ---------------------------------------------------------------------------
# Feature builder — channel ablations
# ---------------------------------------------------------------------------

class TestFeatureBuilder:
    @pytest.fixture
    def raw_features(self):
        # 4 clients, 5 rounds, 7 raw features per round
        rng = np.random.default_rng(0)
        return rng.normal(size=(4, 5, 7)).astype(np.float32)

    @pytest.fixture
    def neighbors(self):
        # 4-client ring
        return [[1, 3], [0, 2], [1, 3], [2, 0]]

    @pytest.fixture
    def omega(self):
        # 2 organizational groups: clients 0,1 in group 0; 2,3 in group 1
        return np.array([0, 0, 1, 1])

    def test_a1_param_only(self, raw_features):
        X = build_attack_features(raw_features, neighbors=None, omega=None, channel="A1")
        # 4 clients × (4 aggregates × 7 raw features) = 28 features
        assert X.shape == (4, 28)

    def test_a2_topo_requires_neighbors(self, raw_features):
        with pytest.raises(ValueError, match="requires neighbors"):
            build_attack_features(raw_features, neighbors=None, omega=None, channel="A2_topo")

    def test_a2_org_requires_omega(self, raw_features, neighbors):
        with pytest.raises(ValueError, match="requires omega"):
            build_attack_features(raw_features, neighbors=neighbors, omega=None, channel="A2_org")

    def test_a2_full_combines(self, raw_features, neighbors, omega):
        X = build_attack_features(raw_features, neighbors=neighbors, omega=omega, channel="A2_full")
        # 28 (params) + 3 (topo: deg, 1/deg, is_central) + 2 (org one-hot for 2 groups) = 33
        assert X.shape == (4, 33)

    def test_a2_topo_only(self, raw_features, neighbors):
        X = build_attack_features(raw_features, neighbors=neighbors, omega=None, channel="A2_topo")
        # No params + 3 topo features
        assert X.shape == (4, 3)

    def test_unknown_channel_raises(self, raw_features):
        with pytest.raises(ValueError, match="Unknown channel"):
            build_attack_features(raw_features, neighbors=None, omega=None, channel="bogus")
