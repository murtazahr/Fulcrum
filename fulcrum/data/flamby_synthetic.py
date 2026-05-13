"""Synthetic re-partitioning of FLamby datasets for TADI shadow training.

The FLamby Fed-ISIC2019 and Fed-Heart-Disease components ship with a fixed
native site partitioning. For *target* experiments we use that partitioning
— it is the deployment realism anchor (Setting A and Setting B in the
manuscript). For *shadow* training, however, we need many runs with
**different** per-client class concentrations $p_i$; otherwise every shadow
run produces the same ground truth and TADI's regressor cannot learn the
relationship between observable features and $p_i$.

This module produces synthetic per-client splits of the FLamby data using
the same Dirichlet($\\alpha$) + position-coupling construction as
:mod:`fulcrum.data.cifar10_eta`. The output is :class:`ClientDataset`
instances, identical in interface to the native FLamby adapters, so the
experiment runner doesn't need to know whether a run is using native or
synthetic partitioning.

Design ref:
    Redesign/03_attack_design.md  §3.4 ("public proxy dataset")
    decisions_log: 2026-05-13 (synthetic-repartitioning commitment)

Caching note: extracting per-sample labels from FLamby's datasets via
iteration is slow (one full augmentation pass per image). Labels are
cached to disk under ``data/_synthetic_cache/`` on first call.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from torch.utils.data import ConcatDataset, Subset

from fulcrum.data import ClientDataset


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FlambySyntheticConfig:
    """Configuration for synthetic re-partitioning of a FLamby dataset.

    Args:
        dataset: which FLamby dataset to re-partition.
        n_clients: number of synthetic clients to produce. The number need
            not match the native FLamby site count; for shadow training
            we typically match the target sweep's client count (6 for
            Setting A, 4 for Setting B) so the regressor's feature width
            matches the target's.
        alpha: Dirichlet concentration parameter. Smaller → more skewed
            per-client class distributions; standard FL non-IID range is
            $\\alpha \\in [0.1, 1.0]$. Default 0.5 gives moderate skew.
        eta: position-coupling weight in $[0, 1]$. 0 = pure Dirichlet
            (IID-null condition); 1 = each client concentrated on class
            $i \\bmod K$. Default 0 so that shadow runs are a clean
            Dirichlet partitioning without engineered position-class
            correlation; the regressor still sees varied $p_i$ across
            seeds because the Dirichlet draws differ per seed.
        sensitive_class: which integer class index is the sensitive
            target $\\mathcal{C}_s$. Setting A: 0 (melanoma). Setting B: 1
            (heart disease positive).
        seed: RNG seed for the Dirichlet draws and shuffles. Each shadow
            run uses a distinct seed so that the resulting $p_i$ vectors
            differ across runs.
        n_classes: number of label classes $K$. Defaults to dataset-specific
            value when None.
        position_smoothing: small uniform mass added to position-target
            distributions to avoid divide-by-zero when $\\eta$ is large.
    """

    dataset: Literal["fed_isic2019", "fed_heart_disease"]
    n_clients: int
    alpha: float = 0.5
    eta: float = 0.0
    sensitive_class: int = 0
    seed: int = 0
    n_classes: int | None = None
    position_smoothing: float = 0.05

    @property
    def resolved_n_classes(self) -> int:
        if self.n_classes is not None:
            return self.n_classes
        return {"fed_isic2019": 8, "fed_heart_disease": 2}[self.dataset]


# ---------------------------------------------------------------------------
# Label extraction (cached)
# ---------------------------------------------------------------------------

def _label_cache_path(data_root: str | Path, dataset: str, split: str) -> Path:
    return Path(data_root) / "_synthetic_cache" / f"{dataset}_labels_{split}.npy"


def _extract_labels(flamby_dataset) -> np.ndarray:
    """Iterate a FLamby dataset once and pull out integer labels.

    Slow path (loads each image through the FLamby augmentation pipeline
    just to read the label) — used only on cache miss. Subsequent calls
    read the cached ``.npy`` file in ~milliseconds.
    """
    labels = []
    for i in range(len(flamby_dataset)):
        _, label = flamby_dataset[i]
        # FLamby returns label as int or 0-dim tensor depending on dataset.
        if hasattr(label, "item"):
            labels.append(int(label.item()))
        else:
            labels.append(int(label))
    return np.asarray(labels, dtype=np.int64)


def _load_labels_cached(
    data_root: str | Path,
    dataset: str,
    flamby_dataset,
    split: str,
) -> np.ndarray:
    cache = _label_cache_path(data_root, dataset, split)
    if cache.exists():
        cached = np.load(cache)
        if cached.shape[0] == len(flamby_dataset):
            return cached
        # Cache stale (dataset size changed) — re-extract.
    labels = _extract_labels(flamby_dataset)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, labels)
    return labels


# ---------------------------------------------------------------------------
# Position-coupled target distributions
# ---------------------------------------------------------------------------

def _position_distributions(n_clients: int, n_classes: int, smoothing: float) -> np.ndarray:
    """Client $i$ targets class $i \\bmod K$, smoothed by a uniform component."""
    base = np.full((n_clients, n_classes), smoothing / n_classes)
    for i in range(n_clients):
        base[i, i % n_classes] += 1.0 - smoothing
    return base


# ---------------------------------------------------------------------------
# Core partitioning logic (shared between A and B)
# ---------------------------------------------------------------------------

def _allocate_indices(
    by_class: dict[int, np.ndarray],
    delta: np.ndarray,
    n_classes: int,
    n_clients: int,
    rng: np.random.Generator,
) -> list[list[int]]:
    """Distribute samples per-class to clients in proportion to delta[:, c].

    Stochastic-rounding allocation so the integer constraint
    $\\sum_i \\text{count}_{c,i} = |\\text{pool}_c|$ holds without
    deterministic bias. Identical algorithm to :func:`cifar10_eta._allocate_indices`.
    """
    assignments: list[list[int]] = [[] for _ in range(n_clients)]
    for c in range(n_classes):
        pool = by_class[c]
        if len(pool) == 0:
            continue
        weights = delta[:, c]
        if weights.sum() <= 0:
            weights = np.ones(n_clients) / n_clients
        weights = weights / weights.sum()
        expected = weights * len(pool)
        floors = np.floor(expected).astype(np.int64)
        residuals = expected - floors
        remaining = len(pool) - int(floors.sum())
        if remaining > 0:
            extras = rng.choice(n_clients, size=remaining, replace=False,
                                p=residuals / residuals.sum())
            for client in extras:
                floors[client] += 1
        cursor = 0
        for client, count in enumerate(floors):
            if count > 0:
                assignments[client].extend(pool[cursor:cursor + count].tolist())
                cursor += count
    return assignments


# ---------------------------------------------------------------------------
# FLamby dataset loaders (returning ConcatDataset across all native centers)
# ---------------------------------------------------------------------------

def _load_pooled_flamby(dataset: str, train: bool):
    """Return a ``ConcatDataset`` of all FLamby centers for the given dataset
    and split. Concatenation order is by center index.
    """
    if dataset == "fed_isic2019":
        from flamby.datasets.fed_isic2019 import FedIsic2019
        centers = [FedIsic2019(center=c, train=train) for c in range(6)]
        return ConcatDataset(centers)
    if dataset == "fed_heart_disease":
        from flamby.datasets.fed_heart_disease import FedHeartDisease
        centers = [FedHeartDisease(center=c, train=train) for c in range(4)]
        return ConcatDataset(centers)
    raise ValueError(f"Unknown FLamby dataset: {dataset!r}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def make_client_datasets(
    cfg: FlambySyntheticConfig,
    data_root: str | Path,
) -> list[ClientDataset]:
    """Materialise a synthetic re-partitioning of a FLamby dataset.

    Returns ``n_clients`` :class:`ClientDataset` instances, each with
    a :class:`~torch.utils.data.Subset` over the pooled FLamby dataset
    and an empirical $\\Delta_i$ / $p_i$ from the actually-allocated
    training labels.

    The same RNG seed produces the same partitioning; vary ``cfg.seed``
    across shadow runs to obtain a distribution of $p_i$ vectors.
    """
    if cfg.n_clients < 1:
        raise ValueError(f"n_clients must be >= 1, got {cfg.n_clients}")
    if not 0.0 <= cfg.eta <= 1.0:
        raise ValueError(f"eta must be in [0, 1], got {cfg.eta}")
    if cfg.alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {cfg.alpha}")

    K = cfg.resolved_n_classes
    rng = np.random.default_rng(cfg.seed)

    # Load pooled FLamby data
    train_pooled = _load_pooled_flamby(cfg.dataset, train=True)
    test_pooled = _load_pooled_flamby(cfg.dataset, train=False)

    # Extract labels (cached on disk to amortise across shadow runs)
    train_labels = _load_labels_cached(data_root, cfg.dataset, train_pooled, "train")
    test_labels = _load_labels_cached(data_root, cfg.dataset, test_pooled, "test")

    # Build per-client target distribution Δ_i
    delta_iid = rng.dirichlet(alpha=np.full(K, cfg.alpha), size=cfg.n_clients)
    if cfg.eta > 0.0:
        delta_pos = _position_distributions(cfg.n_clients, K, cfg.position_smoothing)
        delta = (1.0 - cfg.eta) * delta_iid + cfg.eta * delta_pos
    else:
        delta = delta_iid
    delta = delta / delta.sum(axis=1, keepdims=True)

    # Per-class index pools (shuffled per seed)
    train_by_class = {c: np.where(train_labels == c)[0] for c in range(K)}
    test_by_class = {c: np.where(test_labels == c)[0] for c in range(K)}
    for c in range(K):
        rng.shuffle(train_by_class[c])
        rng.shuffle(test_by_class[c])

    train_assignments = _allocate_indices(train_by_class, delta, K, cfg.n_clients, rng)
    test_assignments = _allocate_indices(test_by_class, delta, K, cfg.n_clients, rng)

    clients: list[ClientDataset] = []
    for i in range(cfg.n_clients):
        train_idx = np.array(train_assignments[i], dtype=np.int64)
        test_idx = np.array(test_assignments[i], dtype=np.int64)
        # Empirical Δ_i from actually-allocated training labels
        if len(train_idx) > 0:
            counts = np.bincount(train_labels[train_idx], minlength=K).astype(np.float64)
            empirical_delta = counts / counts.sum()
        else:
            empirical_delta = delta[i]
        clients.append(ClientDataset(
            client_id=i,
            train_dataset=Subset(train_pooled, train_idx.tolist()),
            test_dataset=Subset(test_pooled, test_idx.tolist()),
            delta=empirical_delta,
            p_sensitive=float(empirical_delta[cfg.sensitive_class]),
            sensitive_class=cfg.sensitive_class,
        ))
    return clients
