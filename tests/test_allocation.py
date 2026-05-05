"""Tests for fulcrum.dp.allocation — Theorem 2 implementation.

These tests verify the mathematical properties guaranteed by the proof:

- KKT balanced condition: a/sigma_i^2 + ell_i is constant across active clients.
- Budget binding: sum_i sigma_i^2 == U at optimum.
- Strict improvement: K* < K_uniform when leverages differ.
- Reduction to uniform when leverages are equal.
- Monotonicity: larger leverage gets larger sigma^2 (more noise).

If any of these break, the math is wrong (the implementation, the proof, or both)
and we want to know about it before running 775 GPU experiments.
"""

from __future__ import annotations

import numpy as np
import pytest

from fulcrum.dp.allocation import (
    optimal_allocation,
    per_client_mi_bound,
    uniform_allocation,
)
from fulcrum.dp.leverage import (
    leverage_degree,
    leverage_eta_position,
    leverage_group_size,
    leverage_uniform,
)


# ---------------------------------------------------------------------------
# Allocation core properties
# ---------------------------------------------------------------------------

class TestAllocationProperties:
    """Properties guaranteed by Theorem 2."""

    def test_uniform_leverages_yield_uniform_allocation(self):
        # When all l_i equal, T2 reduces to sigma_i^2 = U/n.
        n, U, a = 6, 1.0, 0.1
        leverage = leverage_uniform(n, value=2.5)
        alloc = optimal_allocation(leverage, a=a, U=U)
        assert alloc.is_uniform
        np.testing.assert_allclose(alloc.sigma_squared, U / n, rtol=1e-9)

    def test_budget_constraint_binds(self):
        # sum_i sigma_i^2 should equal U exactly (modulo bisection tolerance).
        leverage = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        U, a = 0.5, 0.1
        alloc = optimal_allocation(leverage, a=a, U=U)
        assert alloc.sigma_squared.sum() == pytest.approx(U, rel=1e-6)

    def test_kkt_balanced_per_client_bound(self):
        # At optimum, a/sigma_i^2 + l_i is constant across i and equals K*.
        leverage = np.array([0.5, 1.0, 1.5, 2.0])
        U, a = 0.4, 0.05
        alloc = optimal_allocation(leverage, a=a, U=U)
        per_client = per_client_mi_bound(alloc, leverage, a)
        np.testing.assert_allclose(per_client, alloc.K_star, rtol=1e-6)

    def test_strict_improvement_over_uniform(self):
        # K* < K_uniform when leverage scores are not all equal.
        leverage = np.array([0.1, 0.5, 1.0, 2.0])
        U, a = 0.3, 0.05
        alloc = optimal_allocation(leverage, a=a, U=U)
        assert alloc.K_star < alloc.K_uniform
        # Quantify: the gap should be positive and meaningful (not just numerical)
        assert (alloc.K_uniform - alloc.K_star) > 1e-3

    def test_monotonic_in_leverage(self):
        # Higher leverage should receive more noise (larger sigma^2).
        leverage = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        U, a = 0.6, 0.1
        alloc = optimal_allocation(leverage, a=a, U=U)
        # sigma_i^2 should be monotone increasing with leverage
        assert np.all(np.diff(alloc.sigma_squared) > 0)

    def test_K_star_above_max_leverage(self):
        # K* must exceed max(l_i) for sigma_i^2 = a/(K* - l_i) to be positive.
        leverage = np.array([0.1, 5.0, 10.0])
        U, a = 0.5, 1.0
        alloc = optimal_allocation(leverage, a=a, U=U)
        assert alloc.K_star > leverage.max()
        assert (alloc.sigma_squared > 0).all()

    def test_n_equals_one(self):
        # Single client: sigma^2 = U, K* = a/U + l_0.
        leverage = np.array([1.5])
        U, a = 0.4, 0.2
        alloc = optimal_allocation(leverage, a=a, U=U)
        assert alloc.sigma_squared[0] == pytest.approx(U)
        assert alloc.K_star == pytest.approx(a / U + leverage[0], rel=1e-6)


# ---------------------------------------------------------------------------
# Allocation input validation
# ---------------------------------------------------------------------------

class TestAllocationValidation:
    def test_negative_leverage_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            optimal_allocation(np.array([-0.1, 1.0]), a=0.1, U=0.5)

    def test_non_positive_a_rejected(self):
        with pytest.raises(ValueError, match="a must be"):
            optimal_allocation(np.array([1.0, 2.0]), a=0.0, U=0.5)

    def test_non_positive_U_rejected(self):
        with pytest.raises(ValueError, match="U must be"):
            optimal_allocation(np.array([1.0, 2.0]), a=0.1, U=-0.1)


# ---------------------------------------------------------------------------
# Leverage proxies
# ---------------------------------------------------------------------------

class TestLeverageProxies:
    def test_group_size_proxy(self):
        # omega = [0, 0, 0, 1, 1, 2] -> groups of size 3, 2, 1
        omega = np.array([0, 0, 0, 1, 1, 2])
        ell = leverage_group_size(omega)
        np.testing.assert_array_equal(ell, [3, 3, 3, 2, 2, 1])

    def test_degree_proxy(self):
        # 3-node line: 0-1-2 -> degrees 1, 2, 1
        neighbors = [[1], [0, 2], [1]]
        ell = leverage_degree(neighbors)
        np.testing.assert_array_equal(ell, [1.0, 2.0, 1.0])

    def test_eta_zero_floors_to_epsilon(self):
        # eta=0 -> uniform leverage at floor (avoids degenerate optimization)
        ell = leverage_eta_position(n_clients=5, eta=0.0, n_classes=10)
        assert (ell > 0).all()
        assert ell.std() < 1e-12  # uniform across clients

    def test_eta_one_saturates_at_log_K(self):
        # eta=1 -> leverage scales with log K
        K = 10
        ell = leverage_eta_position(n_clients=5, eta=1.0, n_classes=K)
        assert ell[0] == pytest.approx(np.log(K), rel=1e-9)

    def test_eta_monotone(self):
        # Increasing eta increases leverage
        prev = 0.0
        for eta in [0.0, 0.25, 0.5, 0.75, 1.0]:
            ell = leverage_eta_position(n_clients=4, eta=eta, n_classes=10)
            assert ell[0] >= prev
            prev = float(ell[0])

    def test_eta_position_weights_shape_check(self):
        with pytest.raises(ValueError, match="position_weights must have shape"):
            leverage_eta_position(n_clients=5, eta=0.5, n_classes=10, position_weights=np.ones(3))

    def test_uniform_baseline(self):
        ell = leverage_uniform(7, value=2.0)
        np.testing.assert_array_equal(ell, np.full(7, 2.0))


# ---------------------------------------------------------------------------
# End-to-end: realistic-config sanity check
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_setting_a_hierarchical(self):
        # Setting A: Fed-ISIC2019 with 6 clients in 3 regional groups (2 each)
        omega = np.array([0, 0, 1, 1, 2, 2])
        leverage = leverage_group_size(omega)
        # All groups same size -> uniform leverage -> uniform allocation
        alloc = optimal_allocation(leverage, a=0.1, U=0.6)
        assert alloc.is_uniform
        np.testing.assert_allclose(alloc.sigma_squared, 0.1, rtol=1e-9)

    def test_setting_a_imbalanced_hierarchy(self):
        # Imbalanced: 3 clients in one regional group, 2 in another, 1 alone
        omega = np.array([0, 0, 0, 1, 1, 2])
        leverage = leverage_group_size(omega)
        alloc = optimal_allocation(leverage, a=0.1, U=0.6)
        assert not alloc.is_uniform
        # Higher-leverage clients (in group of 3) should get more noise
        assert alloc.sigma_squared[0] > alloc.sigma_squared[5]
        # Strict improvement
        assert alloc.K_star < alloc.K_uniform

    def test_setting_b_decentralized(self):
        # Setting B: 4-client k-NN proximity graph; degrees vary.
        # Suppose adjacency: 0-1, 0-2, 1-2, 2-3 -> degrees 2, 2, 3, 1
        neighbors = [[1, 2], [0, 2], [0, 1, 3], [2]]
        leverage = leverage_degree(neighbors)
        alloc = optimal_allocation(leverage, a=0.05, U=0.4)
        # Highest-degree client (i=2, deg=3) gets most noise
        assert np.argmax(alloc.sigma_squared) == 2
        # Strict improvement over uniform
        assert alloc.K_star < alloc.K_uniform
