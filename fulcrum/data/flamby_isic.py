"""Setting A — Fed-ISIC2019 adapter.

Wraps FLamby's :class:`FedIsic2019` (6 hospital sites, 8 dermoscopy classes)
into the unified :class:`~fulcrum.data.ClientDataset` interface.

The native site partitioning is what we use; we do not impose any synthetic
re-partitioning here. The empirical class distribution per site is computed
from the loaded labels.

ISIC-2019 class encoding (from FLamby):
    0: MEL  (melanoma)            ← default sensitive class
    1: NV   (nevus)
    2: BCC  (basal cell carcinoma)
    3: AK   (actinic keratosis)
    4: BKL  (benign keratosis)
    5: DF   (dermatofibroma)
    6: VASC (vascular lesion)
    7: SCC  (squamous cell carcinoma)

Design ref: Redesign/02_threat_model_partitioning.md §2.6 Setting A
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fulcrum.data import ClientDataset


SETTING_A_NUM_SITES = 6
SETTING_A_NUM_CLASSES = 8
SETTING_A_DEFAULT_SENSITIVE = 0  # melanoma


@dataclass(frozen=True)
class FedISICConfig:
    """Configuration for the Fed-ISIC2019 adapter."""

    sensitive_class: int = SETTING_A_DEFAULT_SENSITIVE
    pooled: bool = False  # always False for federated training; True only for centralized baselines


def _label_array_from_dataset(dataset) -> np.ndarray:
    """Extract integer labels from a FLamby Fed-ISIC2019 dataset.

    FLamby's dataset yields ``(image_tensor, label_tensor)`` per index. We iterate
    once to materialize the label array. This is cheap because the dataset is
    small (~10K total across 6 sites) and we only read the labels.
    """
    labels = []
    for i in range(len(dataset)):
        _, y = dataset[i]
        if hasattr(y, "item"):
            labels.append(int(y.item()))
        else:
            labels.append(int(y))
    return np.asarray(labels, dtype=np.int64)


def make_client_datasets(cfg: FedISICConfig | None = None) -> list[ClientDataset]:
    """Construct the 6-site Fed-ISIC2019 federation.

    Args:
        cfg: adapter config; defaults to melanoma as sensitive class.

    Returns:
        Length-6 list of :class:`ClientDataset` (one per hospital site), with
        empirical $\\Delta_i$ and $p_i$ computed from the actual class
        distribution at each site.

    Raises:
        ImportError: if FLamby is not installed (run ``scripts/setup_env.sh``).
        FileNotFoundError: if Fed-ISIC2019 has not been downloaded yet
            (run ``scripts/download_data.py`` and follow the manual ISIC
            instructions if needed).
    """
    if cfg is None:
        cfg = FedISICConfig()
    if not 0 <= cfg.sensitive_class < SETTING_A_NUM_CLASSES:
        raise ValueError(
            f"sensitive_class must be in [0, {SETTING_A_NUM_CLASSES}), got {cfg.sensitive_class}"
        )

    try:
        from flamby.datasets.fed_isic2019 import FedIsic2019
    except ImportError as exc:
        raise ImportError(
            "FLamby's Fed-ISIC2019 dataset is not installed. Run scripts/setup_env.sh."
        ) from exc

    clients: list[ClientDataset] = []
    for center in range(SETTING_A_NUM_SITES):
        train_ds = FedIsic2019(center=center, train=True, pooled=cfg.pooled)
        test_ds = FedIsic2019(center=center, train=False, pooled=cfg.pooled)

        train_labels = _label_array_from_dataset(train_ds)
        if len(train_labels) == 0:
            delta = np.full(SETTING_A_NUM_CLASSES, 1.0 / SETTING_A_NUM_CLASSES)
        else:
            counts = np.bincount(train_labels, minlength=SETTING_A_NUM_CLASSES).astype(np.float64)
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


def default_organizational_labels() -> np.ndarray:
    """Default $\\omega$ for Setting A: 3 regional aggregators, 2 sites each.

    This implements the hierarchical assignment used in the canonical Setting A
    config: sites 0,1 → region 0; sites 2,3 → region 1; sites 4,5 → region 2.
    """
    return np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
