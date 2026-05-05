"""Structural-leverage proxies $\\ell_i^\\circ$ from $(\\mathcal{G}, \\omega)$.

Three concrete proxies, one per setting:

- :func:`leverage_group_size` — Setting A (hierarchical FL).
  Corollary 1 (SBM): $\\ell_i^\\circ \\propto |G_i|$ where $G_i = \\omega^{-1}(\\omega_i)$.
  Tight in the small-block regime ($|G_i|\\,\\kappa < H(\\Phi)$); loose otherwise.

- :func:`leverage_degree` — Setting B (decentralized k-NN proximity).
  Corollary 2 (heuristic): $\\ell_i^\\circ \\propto \\deg_{\\mathcal{G}}(i)$.
  Empirically validated, not provably tight — see design doc §4.6.6.

- :func:`leverage_eta_position` — Setting C (synthetic).
  $\\ell_i^\\circ = c \\cdot \\eta$ scaled per-position; reflects that lateral
  leakage scales with the topology-data coupling parameter $\\eta$.

All functions return non-negative leverage scores. Absolute scale is irrelevant
to :func:`fulcrum.dp.allocation.optimal_allocation` — only relative magnitudes
matter for the min-max optimization.

The :func:`leverage_uniform` baseline returns a flat vector; passing it to
:func:`optimal_allocation` collapses Theorem 2 to standard uniform DP-SGD.
This is the comparator for the Pareto frontier.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Setting A — organizational group size (Corollary 1)
# ---------------------------------------------------------------------------

def leverage_group_size(omega: np.ndarray) -> np.ndarray:
    """Per-client leverage = size of organizational group $|G_i|$.

    Args:
        omega: integer array of length $n$ mapping client $i$ to its group label.

    Returns:
        Length-$n$ array $\\ell$ where $\\ell_i = |\\{j : \\omega_j = \\omega_i\\}|$.
    """
    omega = np.asarray(omega, dtype=np.int64)
    if omega.ndim != 1:
        raise ValueError(f"omega must be 1D, got shape {omega.shape}")
    counts = np.bincount(omega)
    return counts[omega].astype(np.float64)


# ---------------------------------------------------------------------------
# Setting B — graph degree (Corollary 2 heuristic)
# ---------------------------------------------------------------------------

def leverage_degree(neighbors: list[list[int]]) -> np.ndarray:
    """Per-client leverage = node degree in the topology.

    Args:
        neighbors: adjacency list (Murmura ``Topology.neighbors`` format).

    Returns:
        Length-$n$ array of degrees as floats.
    """
    return np.array([len(nbrs) for nbrs in neighbors], dtype=np.float64)


# ---------------------------------------------------------------------------
# Setting C — synthetic, scales with the coupling parameter η
# ---------------------------------------------------------------------------

def leverage_eta_position(
    n_clients: int,
    eta: float,
    n_classes: int,
    position_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Per-client leverage as a function of $\\eta$ and topology position.

    The η-coupled partitioning (see :mod:`fulcrum.data.cifar10_eta`) makes the
    lateral leakage $I(p_i; D_{-i})$ scale with $\\eta$. At $\\eta = 0$ leverage
    is zero (IID); at $\\eta = 1$ it saturates near $\\log K$. Per-client
    asymmetries (when topology positions are non-equivalent) are captured via
    optional ``position_weights``.

    Args:
        n_clients: number of clients $n$.
        eta: coupling parameter in $[0, 1]$.
        n_classes: number of classes $K$ (used for the saturation cap).
        position_weights: optional length-$n$ array of relative position weights;
            defaults to uniform ones.

    Returns:
        Length-$n$ leverage vector. A floor of ``1e-9`` is applied to avoid
        zero leverage degenerating the allocation optimization.
    """
    if not 0.0 <= eta <= 1.0:
        raise ValueError(f"eta must be in [0, 1], got {eta}")
    if position_weights is None:
        position_weights = np.ones(n_clients, dtype=np.float64)
    else:
        position_weights = np.asarray(position_weights, dtype=np.float64)
        if position_weights.shape != (n_clients,):
            raise ValueError(
                f"position_weights must have shape ({n_clients},), got {position_weights.shape}"
            )

    base = eta * np.log(max(n_classes, 2))
    return np.maximum(base * position_weights, 1e-9)


# ---------------------------------------------------------------------------
# Settings A and B — dataset-size weighted leverage (FedAvg-influence heuristic)
# ---------------------------------------------------------------------------

def leverage_dataset_size(dataset_sizes) -> np.ndarray:
    """Per-client leverage proportional to training-set size, mean-normalised.

    **Motivation.** Standard FedAvg aggregates client updates with weights
    $w_i = |\\mathcal{D}_i| / \\sum_j |\\mathcal{D}_j|$. A client with more
    training samples contributes proportionally more to the global model, and
    its data therefore has correspondingly more influence on what the adversary
    observes through $\\Theta$. This is a heuristic complement to the
    SBM-derived :func:`leverage_group_size` (Corollary 1) and degree-based
    :func:`leverage_degree` (Corollary 2 heuristic) — applicable when topology
    is symmetric (uniform degree) but client dataset sizes vary substantially,
    which is the typical case in FLamby's natively-partitioned cross-silo
    benchmarks.

    For Setting A (Fed-ISIC2019, 6 sites with sizes ~[9930, 3163, 2691, 1807,
    655, 351]) this gives leverage ratios of ~29× across clients; for Setting B
    (Fed-Heart-Disease, 4 sites with [199, 172, 30, 85]) it gives ~6.6×. Both
    produce non-degenerate Theorem 2 allocations even when topology is uniform.

    Args:
        dataset_sizes: per-client training set sizes (length-$n$ array-like).

    Returns:
        Length-$n$ array of leverage values normalised to mean 1. Floors at
        ``1e-9`` to avoid degenerate optimization if any client has 0 samples.
    """
    sizes = np.asarray(dataset_sizes, dtype=np.float64)
    if sizes.ndim != 1:
        raise ValueError(f"dataset_sizes must be 1D, got shape {sizes.shape}")
    if (sizes < 0).any():
        raise ValueError("dataset_sizes must be non-negative")
    mean = sizes.mean()
    if mean <= 0:
        return np.full(sizes.size, 1e-9)
    return np.maximum(sizes / mean, 1e-9)


# ---------------------------------------------------------------------------
# Baseline — uniform leverage (collapses Theorem 2 to uniform allocation)
# ---------------------------------------------------------------------------

def leverage_uniform(n_clients: int, value: float = 1.0) -> np.ndarray:
    """Flat leverage vector. Use as the baseline against the topology-aware allocation."""
    return np.full(n_clients, value, dtype=np.float64)


# ---------------------------------------------------------------------------
# Validation helper — fails loudly if leverage is uniform under topology_aware
# ---------------------------------------------------------------------------

def assert_non_uniform_leverage(leverage: np.ndarray, allocation_mode: str, proxy: str) -> None:
    """Raise if leverage is constant when allocation mode is ``topology_aware``.

    The combination of constant leverage + topology_aware allocation produces
    Theorem 2's reduction case (uniform allocation), making the experiment
    silently degenerate. This check catches that at config-validation time
    before training starts, instead of revealing it in the metrics.json after
    hours of compute.
    """
    if allocation_mode != "topology_aware":
        return
    leverage = np.asarray(leverage, dtype=np.float64)
    if leverage.size <= 1:
        return
    if np.allclose(leverage, leverage[0], rtol=1e-9, atol=1e-12):
        raise ValueError(
            f"Leverage proxy '{proxy}' produced uniform values "
            f"(all = {leverage[0]:.6f}) under allocation='topology_aware'. "
            "Theorem 2 will collapse to uniform allocation, making the "
            "experiment degenerate. Either change the leverage proxy "
            "(e.g., 'dataset_size' for cross-silo settings, 'group_size' for "
            "hierarchical with uneven groups), change the topology to break "
            "degree symmetry, or set allocation='uniform' explicitly."
        )
