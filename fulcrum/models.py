"""Per-setting model factories.

Three small, Opacus-compatible architectures — one per setting:

- Setting A (Fed-ISIC2019, 224×224×3 dermoscopy, 8 classes): small CNN with
  GroupNorm (BatchNorm is incompatible with per-sample gradients).
- Setting B (Fed-Heart-Disease, 13 features, binary): logistic regression with
  optional hidden layer.
- Setting C (CIFAR-10, 32×32×3, 10 classes): small CNN with GroupNorm.

Models are kept small to fit the L40S compute budget. ResNet-18 on Setting A
would be more accurate but ~5–10× slower per round; the ablation between
"small CNN" and "ResNet" is a follow-up.

Opacus compatibility constraints honored throughout:
- GroupNorm instead of BatchNorm (per-sample gradients require it).
- ``nn.ReLU()`` *without* ``inplace=True`` — in-place activations break
  Opacus's autograd hooks ("Output 0 of BackwardHookFunctionBackward is a
  view and is being modified inplace"). Same applies to any other in-place op
  if added later (``x += ...``, ``F.relu(x, inplace=True)``, etc.).
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn


SettingName = Literal["A", "B", "C"]


def make_model(setting: SettingName) -> nn.Module:
    """Return the canonical model for the given setting.

    Args:
        setting: ``"A"``, ``"B"``, or ``"C"``.

    Returns:
        ``torch.nn.Module`` that takes setting-appropriate input and returns logits.
    """
    if setting == "A":
        return _make_isic_cnn()
    if setting == "B":
        return _make_heart_mlp()
    if setting == "C":
        return _make_cifar_cnn()
    raise ValueError(f"Unknown setting: {setting!r}")


# ---------------------------------------------------------------------------
# Setting A — Fed-ISIC2019 (224×224×3 → 8 classes)
# ---------------------------------------------------------------------------

def _gn(num_channels: int, num_groups: int = 8) -> nn.Module:
    """GroupNorm with safe num_groups selection."""
    return nn.GroupNorm(num_groups=min(num_groups, num_channels), num_channels=num_channels)


def _make_isic_cnn() -> nn.Module:
    """Compact CNN for 224×224 dermoscopy images, 8 classes."""
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),  # 112×112
        _gn(32),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),       # 56×56
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 28×28
        _gn(64),
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 14×14
        _gn(128),
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),# 7×7
        _gn(128),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(128, 8),
    )


# ---------------------------------------------------------------------------
# Setting B — Fed-Heart-Disease (13 features → 2 classes)
# ---------------------------------------------------------------------------

def _make_heart_mlp(input_dim: int = 13, hidden: int = 32) -> nn.Module:
    """Small MLP for tabular heart disease prediction."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 2),
    )


# ---------------------------------------------------------------------------
# Setting C — CIFAR-10 (32×32×3 → 10 classes)
# ---------------------------------------------------------------------------

def _make_cifar_cnn() -> nn.Module:
    """Compact CNN for CIFAR-10."""
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        _gn(32),
        nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size=3, padding=1),
        _gn(32),
        nn.ReLU(),
        nn.MaxPool2d(2),                    # 16×16
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        _gn(64),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        _gn(64),
        nn.ReLU(),
        nn.MaxPool2d(2),                    # 8×8
        nn.AdaptiveAvgPool2d((4, 4)),
        nn.Flatten(),
        nn.Linear(64 * 16, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
