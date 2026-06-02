"""Setting B — Fed-Heart-Disease adapter.

Wraps FLamby's :class:`FedHeartDisease` (4 hospital sites, binary classification)
into the unified :class:`~fulcrum.data.ClientDataset` interface.

Setting B is binary, so the inference target reduces to per-site positive-rate
prediction (Stage 2 design adjustment). The "sensitive class" is positive heart
disease (label 1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fulcrum.data import ClientDataset


SETTING_B_NUM_SITES = 4
SETTING_B_NUM_CLASSES = 2
SETTING_B_DEFAULT_SENSITIVE = 1  # positive heart disease


@dataclass(frozen=True)
class FedHeartConfig:
    """Configuration for the Fed-Heart-Disease adapter."""

    sensitive_class: int = SETTING_B_DEFAULT_SENSITIVE
    pooled: bool = False


def _label_array_from_dataset(dataset) -> np.ndarray:
    """Extract integer labels from a FLamby Fed-Heart-Disease dataset."""
    labels = []
    for i in range(len(dataset)):
        _, y = dataset[i]
        if hasattr(y, "item"):
            labels.append(int(y.item()))
        else:
            labels.append(int(y))
    return np.asarray(labels, dtype=np.int64)


def make_client_datasets(cfg: FedHeartConfig | None = None) -> list[ClientDataset]:
    """Construct the 4-site Fed-Heart-Disease federation.

    Returns length-4 list of :class:`ClientDataset` with empirical positive-rate
    per site as ``p_sensitive``.
    """
    if cfg is None:
        cfg = FedHeartConfig()
    if not 0 <= cfg.sensitive_class < SETTING_B_NUM_CLASSES:
        raise ValueError(
            f"sensitive_class must be in [0, {SETTING_B_NUM_CLASSES}), got {cfg.sensitive_class}"
        )

    try:
        from flamby.datasets.fed_heart_disease import FedHeartDisease
    except ImportError as exc:
        raise ImportError(
            "FLamby's Fed-Heart-Disease dataset is not installed. Run scripts/setup_env.sh."
        ) from exc

    clients: list[ClientDataset] = []
    for center in range(SETTING_B_NUM_SITES):
        train_ds = FedHeartDisease(center=center, train=True, pooled=cfg.pooled)
        test_ds = FedHeartDisease(center=center, train=False, pooled=cfg.pooled)

        train_labels = _label_array_from_dataset(train_ds)
        if len(train_labels) == 0:
            delta = np.full(SETTING_B_NUM_CLASSES, 1.0 / SETTING_B_NUM_CLASSES)
        else:
            counts = np.bincount(train_labels, minlength=SETTING_B_NUM_CLASSES).astype(np.float64)
            delta = counts / counts.sum()

        clients.append(
            ClientDataset(
                client_id=center,
                train_dataset=train_ds,
                test_dataset=test_ds,
                delta=delta,
                p_sensitive=float(delta[cfg.sensitive_class]),
                sensitive_class=cfg.sensitive_class,
            )
        )
    return clients
