"""Topology-aware DP-SGD noise allocation — Theorem 2 implementation.

Solves the constrained min-max problem
    minimize   max_i [ a / sigma_i^2 + ell_i ]
    subject to sum_i sigma_i^2 <= U,    sigma_i^2 > 0
via the closed-form KKT solution
    sigma_i^{*2} = a / (K* - ell_i)
where $K^\\star > \\max_i \\ell_i$ is the unique solution of the budget equation
    g(K) := sum_i a / (K - ell_i) = U.

$g$ is strictly decreasing on $K > \\max_i \\ell_i$ with $g \\to +\\infty$ at
the boundary and $g \\to 0$ at infinity, so 1D bisection converges in
$O(\\log(1/\\text{tol}))$ iterations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Allocation:
    """Result of the optimal allocation.

    Attributes:
        sigma_squared: per-client noise variances $\\sigma_i^{*2}$.
        K_star: the worst-case MI bound — by Theorem 2 the value is balanced
            across all clients at $K^\\star$.
        K_uniform: the worst-case MI bound under uniform allocation
            $\\sigma_i^2 = U/n$, included for the strict-improvement comparator.
        budget: the utility budget $U$ (for sanity checking).
        is_uniform: True if all leverage scores were equal (T2 collapses to uniform).
    """

    sigma_squared: np.ndarray
    K_star: float
    K_uniform: float
    budget: float
    is_uniform: bool

    def sigma(self) -> np.ndarray:
        """Per-client noise std (square root of variance)."""
        return np.sqrt(self.sigma_squared)


def optimal_allocation(
    leverage: np.ndarray,
    a: float,
    U: float,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> Allocation:
    """Compute the topology-aware DP noise allocation per Theorem 2.

    Args:
        leverage: non-negative per-client leverage scores $\\ell_i^\\circ$.
        a: DP-SGD coefficient $a = T_{\\max} C^2 / (2 |B|^2)$ — the per-round per-coordinate
            KL bound for the Gaussian mechanism with sensitivity $C/|B|$ at unit noise.
        U: utility budget $\\sum_i \\sigma_i^2 \\leq U$.
        tol: bisection tolerance on $K$.
        max_iter: bisection iteration cap.

    Returns:
        Allocation with the optimal per-client variances and the achieved $K^\\star$.

    Raises:
        ValueError: if inputs are out of range.
    """
    leverage = np.asarray(leverage, dtype=np.float64)
    if leverage.ndim != 1:
        raise ValueError(f"leverage must be 1D, got shape {leverage.shape}")
    if (leverage < 0).any():
        raise ValueError("leverage entries must be non-negative")
    if a <= 0:
        raise ValueError(f"a must be > 0, got {a}")
    if U <= 0:
        raise ValueError(f"U must be > 0, got {U}")

    n = leverage.size
    ell_max = float(leverage.max())
    K_uniform = a * n / U + ell_max

    # Degenerate case: all leverages equal -> uniform allocation, no bisection needed.
    if np.allclose(leverage, leverage[0], atol=1e-12):
        sigma_sq = np.full(n, U / n)
        return Allocation(
            sigma_squared=sigma_sq,
            K_star=K_uniform,
            K_uniform=K_uniform,
            budget=U,
            is_uniform=True,
        )

    # Bisect on K in (ell_max, K_uniform]. By optimality K* <= K_uniform.
    # g is strictly decreasing; we want g(K*) = U.
    lo = ell_max + max(1e-12, 1e-9 * max(ell_max, 1.0))
    hi = K_uniform

    def g(K: float) -> float:
        return float(np.sum(a / (K - leverage)))

    # Sanity: g(lo) should be huge, g(hi) should be <= U
    assert g(hi) <= U + tol, f"Internal error: g(K_uniform)={g(hi)} > U={U}"

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if hi - lo < tol * max(1.0, abs(hi)):
            break
        if g(mid) > U:
            lo = mid
        else:
            hi = mid
    K_star = 0.5 * (lo + hi)
    sigma_sq = a / (K_star - leverage)

    return Allocation(
        sigma_squared=sigma_sq,
        K_star=float(K_star),
        K_uniform=K_uniform,
        budget=U,
        is_uniform=False,
    )


def uniform_allocation(n_clients: int, U: float, leverage: np.ndarray, a: float) -> Allocation:
    """Uniform $\\sigma_i^2 = U/n$ allocation — the comparator baseline.

    Returns the same Allocation type so downstream analysis treats topology-aware
    and uniform identically.
    """
    if U <= 0 or a <= 0 or n_clients < 1:
        raise ValueError("Invalid arguments to uniform_allocation")
    leverage = np.asarray(leverage, dtype=np.float64)
    sigma_sq = np.full(n_clients, U / n_clients)
    K_uniform = a * n_clients / U + float(leverage.max())
    return Allocation(
        sigma_squared=sigma_sq,
        K_star=K_uniform,
        K_uniform=K_uniform,
        budget=U,
        is_uniform=True,
    )


def per_client_mi_bound(allocation: Allocation, leverage: np.ndarray, a: float) -> np.ndarray:
    """Theorem 1's per-client MI bound: $a / \\sigma_i^2 + \\ell_i^\\circ$.

    Useful for diagnostics — at the optimal allocation this should be uniform across
    clients and equal to ``allocation.K_star``.
    """
    leverage = np.asarray(leverage, dtype=np.float64)
    return a / allocation.sigma_squared + leverage
