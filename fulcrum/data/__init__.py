"""Per-client dataset types shared across all setting adapters.

Each setting adapter (``cifar10_eta``, ``flamby_isic``, ``flamby_heart``)
returns a list of :class:`ClientDataset` — the same uniform interface so the
experiment runner doesn't need to know which dataset it is operating on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from torch.utils.data import Dataset


@dataclass
class ClientDataset:
    """Materialized per-client data + ground-truth metadata.

    Attributes:
        client_id: integer client index (0..n-1).
        train_dataset: torch ``Dataset`` for training. Each adapter returns
            a Subset, FLamby per-center dataset, etc. — but always a torch
            Dataset that yields ``(input, label)`` pairs.
        test_dataset: torch ``Dataset`` for evaluation.
        delta: empirical class distribution, length $K$. Sum to 1.
        p_sensitive: $p_i = \\Delta_i(\\mathcal{C}_s)$ — the inference target.
        sensitive_class: which class index counts as "sensitive" for this setting.
    """

    client_id: int
    train_dataset: "Dataset"
    test_dataset: "Dataset"
    delta: np.ndarray
    p_sensitive: float
    sensitive_class: int

    @property
    def n_train(self) -> int:
        return len(self.train_dataset)  # type: ignore[arg-type]

    @property
    def n_test(self) -> int:
        return len(self.test_dataset)  # type: ignore[arg-type]


def ground_truth_p(clients: list[ClientDataset]) -> np.ndarray:
    """Vector of true sensitive-class concentrations $p_i$ across all clients."""
    return np.array([c.p_sensitive for c in clients], dtype=np.float64)


def organizational_labels(clients: list[ClientDataset]) -> np.ndarray:
    """Default $\\omega$ for adapters that don't supply one (each client = own group)."""
    return np.arange(len(clients), dtype=np.int64)
