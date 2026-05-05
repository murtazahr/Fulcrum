"""CIFAR-10 partitioning with Dirichlet heterogeneity + topology coupling η.

This is Setting C from the experimental design. See:
    Redesign/02_threat_model_partitioning.md  §2.6 Setting C
    Redesign/04_defense_design.md             §4.4 leverage definition

Per-client class distribution Δ_i is constructed as:

    Δ_i = (1 - η) · Δ_iid + η · Δ_pos(φ_i)

where:
    Δ_iid ~ Dirichlet(α)               (IID heterogeneity component)
    Δ_pos(φ_i)                         (position-determined target — concentrated
                                        on class φ_i mod K)
    η ∈ [0, 1]                         (topology-data coupling parameter)
    φ_i                                (client i's topology position index)

η = 0 ⇒ pure Dirichlet, no position correlation, the IID null condition.
η = 1 ⇒ Δ_i is fully determined by topology position.

Returns per-client (indices, labels) splits suitable for federated training.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as T


# Standard CIFAR-10 normalization stats — same values used in essentially every
# CIFAR-10 paper since He et al. 2016. ToTensor converts PIL Image (HWC uint8)
# to tensor (CHW float in [0, 1]); Normalize scales to ~zero-mean / unit-std.
# We keep transforms deterministic (no augmentation) for clean per-sample DP
# sensitivity bookkeeping and to keep TADI features stable across rounds.
_CIFAR10_TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
])


@dataclass(frozen=True)
class PartitionConfig:
    """Configuration for the η-coupled CIFAR-10 partitioning."""

    n_clients: int
    eta: float                  # in [0, 1]
    alpha: float                # Dirichlet concentration (smaller = more skew)
    seed: int = 0
    n_classes: int = 10
    sensitive_class: int = 0    # designated sensitive class C_s (default: airplane)
    position_smoothing: float = 0.05  # weight on uniform when building Δ_pos to avoid divide-by-zero


@dataclass(frozen=True)
class ClientPartition:
    """Per-client data partition."""

    train_indices: np.ndarray            # int indices into the source dataset
    test_indices: np.ndarray             # int indices into the source test set (shared evaluation)
    delta: np.ndarray                    # length-K class distribution at this client
    p_sensitive: float                   # = delta[sensitive_class]


def make_position_distributions(n_clients: int, n_classes: int, smoothing: float) -> np.ndarray:
    """Build position-determined target distributions Δ_pos.

    Client i's target distribution is concentrated on class (i mod n_classes),
    smoothed by a small uniform component to keep all classes represented.

    Returns array of shape (n_clients, n_classes), each row is a probability vector.
    """
    base = np.full((n_clients, n_classes), smoothing / n_classes)
    for i in range(n_clients):
        base[i, i % n_classes] += 1.0 - smoothing
    return base


def make_partition(cfg: PartitionConfig, dataset_root: str | Path) -> list[ClientPartition]:
    """Build the η-coupled per-client partition over CIFAR-10."""
    if not 0.0 <= cfg.eta <= 1.0:
        raise ValueError(f"eta must be in [0, 1], got {cfg.eta}")
    if cfg.alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {cfg.alpha}")
    if cfg.n_clients < 1:
        raise ValueError(f"n_clients must be >= 1, got {cfg.n_clients}")

    rng = np.random.default_rng(cfg.seed)

    # Load CIFAR-10 (expects download already completed by scripts/download_data.py).
    # download_data.py places CIFAR-10 under <data_root>/cifar10/, so we append
    # that suffix here for torchvision's root argument.
    cifar_root = Path(dataset_root) / "cifar10"
    train = torchvision.datasets.CIFAR10(
        root=str(cifar_root), train=True, download=False, transform=_CIFAR10_TRANSFORM,
    )
    test = torchvision.datasets.CIFAR10(
        root=str(cifar_root), train=False, download=False, transform=_CIFAR10_TRANSFORM,
    )
    train_targets = np.asarray(train.targets, dtype=np.int64)
    test_targets = np.asarray(test.targets, dtype=np.int64)

    K = cfg.n_classes

    # Build per-client target distributions
    # Δ_iid_i ~ Dirichlet(α · 1_K)  for each client i
    delta_iid = rng.dirichlet(alpha=np.full(K, cfg.alpha), size=cfg.n_clients)
    # Δ_pos_i: position-determined targets
    delta_pos = make_position_distributions(cfg.n_clients, K, cfg.position_smoothing)
    # Δ_i = (1 - η) Δ_iid + η Δ_pos
    delta = (1.0 - cfg.eta) * delta_iid + cfg.eta * delta_pos
    # Numerical safety: renormalise
    delta = delta / delta.sum(axis=1, keepdims=True)

    # Per-class index pools
    train_by_class = {c: np.where(train_targets == c)[0] for c in range(K)}
    test_by_class = {c: np.where(test_targets == c)[0] for c in range(K)}
    for c in range(K):
        rng.shuffle(train_by_class[c])
        rng.shuffle(test_by_class[c])

    # Allocate samples to clients proportionally to delta[i, c]
    # Train: split each class's samples across clients in proportion to delta[:, c]
    train_assignments = _allocate_indices(train_by_class, delta, K, cfg.n_clients, rng)
    test_assignments = _allocate_indices(test_by_class, delta, K, cfg.n_clients, rng)

    partitions: list[ClientPartition] = []
    for i in range(cfg.n_clients):
        train_idx = np.array(train_assignments[i], dtype=np.int64)
        test_idx = np.array(test_assignments[i], dtype=np.int64)
        # Empirical Δ from the actually-allocated training labels (may differ slightly from target)
        if len(train_idx) > 0:
            counts = np.bincount(train_targets[train_idx], minlength=K).astype(np.float64)
            empirical_delta = counts / counts.sum()
        else:
            empirical_delta = delta[i]
        partitions.append(
            ClientPartition(
                train_indices=train_idx,
                test_indices=test_idx,
                delta=empirical_delta,
                p_sensitive=float(empirical_delta[cfg.sensitive_class]),
            )
        )
    return partitions


def _allocate_indices(
    by_class: dict[int, np.ndarray],
    delta: np.ndarray,
    n_classes: int,
    n_clients: int,
    rng: np.random.Generator,
) -> list[list[int]]:
    """Distribute samples per-class to clients in proportion to delta[:, c].

    Uses a stochastic rounding step to handle the integer constraint without
    deterministic bias.
    """
    assignments: list[list[int]] = [[] for _ in range(n_clients)]
    for c in range(n_classes):
        pool = by_class[c]
        if len(pool) == 0:
            continue
        weights = delta[:, c]
        if weights.sum() <= 0:
            # Degenerate; share uniformly
            weights = np.ones(n_clients) / n_clients
        weights = weights / weights.sum()
        # Expected count per client; use stochastic rounding to integers summing to len(pool)
        expected = weights * len(pool)
        floors = np.floor(expected).astype(np.int64)
        residuals = expected - floors
        remaining = len(pool) - int(floors.sum())
        if remaining > 0:
            extras = rng.choice(n_clients, size=remaining, replace=False, p=residuals / residuals.sum())
            for client in extras:
                floors[client] += 1
        # Slice the pool sequentially per client
        cursor = 0
        for client, count in enumerate(floors):
            if count > 0:
                assignments[client].extend(pool[cursor:cursor + count].tolist())
                cursor += count
    return assignments


# ---------------------------------------------------------------------------
# Convenience: ground truth p_i extraction for evaluation
# ---------------------------------------------------------------------------

def ground_truth_p(partitions: list[ClientPartition]) -> np.ndarray:
    """Vector of true sensitive-class concentrations p_i across clients."""
    return np.array([part.p_sensitive for part in partitions], dtype=np.float64)


# ---------------------------------------------------------------------------
# Unified ClientDataset interface (matches FLamby adapters)
# ---------------------------------------------------------------------------

def make_client_datasets(cfg: PartitionConfig, dataset_root: str | Path):
    """Materialize the η-coupled partition into ``ClientDataset`` objects.

    Same interface as :func:`fulcrum.data.flamby_isic.make_client_datasets` and
    :func:`fulcrum.data.flamby_heart.make_client_datasets` — the experiment runner
    can treat all three settings uniformly.

    Returns:
        List of :class:`~fulcrum.data.ClientDataset`. Each client gets a torch
        :class:`~torch.utils.data.Subset` over the global CIFAR-10 train/test
        sets, plus the empirical $\\Delta_i$ and $p_i$ from :func:`make_partition`.
    """
    from torch.utils.data import Subset
    from fulcrum.data import ClientDataset

    partitions = make_partition(cfg, dataset_root)

    # Reload the underlying torchvision datasets once to wrap with Subset.
    # Same <data_root>/cifar10/ subdir convention as make_partition.
    cifar_root = Path(dataset_root) / "cifar10"
    train_full = torchvision.datasets.CIFAR10(
        root=str(cifar_root), train=True, download=False, transform=_CIFAR10_TRANSFORM,
    )
    test_full = torchvision.datasets.CIFAR10(
        root=str(cifar_root), train=False, download=False, transform=_CIFAR10_TRANSFORM,
    )

    clients: list[ClientDataset] = []
    for i, part in enumerate(partitions):
        clients.append(
            ClientDataset(
                client_id=i,
                train_dataset=Subset(train_full, part.train_indices.tolist()),
                test_dataset=Subset(test_full, part.test_indices.tolist()),
                delta=part.delta,
                p_sensitive=part.p_sensitive,
                sensitive_class=cfg.sensitive_class,
            )
        )
    return clients
