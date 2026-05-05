"""Per-client Opacus DP-SGD wrapper.

Opacus's :class:`~opacus.PrivacyEngine` is normally configured with a single
``noise_multiplier`` shared across all participants. Our defense (D1) requires
per-client noise scales $\\{\\sigma_i\\}$ derived from
:func:`fulcrum.dp.allocation.optimal_allocation`. This module provides a
per-client wrapper that creates a private (model, optimizer, dataloader)
triple with the correct $\\sigma_i$ for each client.

It also handles the standard Opacus model-validity step
(:class:`~opacus.validators.ModuleValidator.fix`) which replaces incompatible
modules (BatchNorm, LayerNorm in some forms, etc.) with DP-friendly variants.

See:
    Redesign/04_defense_design.md  §4.6.5 (Theorem 1 proof, Lemma 2)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


@dataclass
class PrivateClientHandles:
    """Handles to a per-client privatized training pipeline.

    Attributes:
        model: the (possibly module-fixed) DP-wrapped model.
        optimizer: the DPOptimizer wrapping the original optimizer.
        dataloader: the DPDataLoader wrapping the original loader.
        privacy_engine: the per-client :class:`~opacus.PrivacyEngine` instance
            (kept for accounting introspection).
        noise_multiplier: the $\\sigma_i$ value used.
        max_grad_norm: the clipping bound $C$ used.
    """

    model: nn.Module
    optimizer: Optimizer
    dataloader: DataLoader
    privacy_engine: Any
    noise_multiplier: float
    max_grad_norm: float


def make_private_client(
    model: nn.Module,
    optimizer: Optimizer,
    dataloader: DataLoader,
    noise_multiplier: float,
    max_grad_norm: float = 1.0,
    poisson_sampling: bool = True,
) -> PrivateClientHandles:
    """Wrap a single client's (model, optimizer, dataloader) in Opacus DP-SGD.

    Args:
        model: client's local model. If incompatible with Opacus, it is fixed
            via :meth:`opacus.validators.ModuleValidator.fix` (in place where possible).
        optimizer: client's optimizer (will be wrapped as a DPOptimizer).
        dataloader: client's training dataloader.
        noise_multiplier: per-client $\\sigma_i$ from the allocation.
        max_grad_norm: clipping bound $C$ (default 1.0).
        poisson_sampling: use Poisson per-record subsampling (Opacus default; required
            for the standard DP-SGD privacy guarantee). Set False to use the dataloader's
            existing sampler unchanged (requires the dataloader to already implement
            an appropriate sampler).

    Returns:
        :class:`PrivateClientHandles` with the privatized triple.

    Raises:
        ImportError: if Opacus is not installed.
        ValueError: if ``noise_multiplier`` is non-positive.
    """
    if noise_multiplier <= 0:
        raise ValueError(f"noise_multiplier must be > 0, got {noise_multiplier}")

    try:
        from opacus import PrivacyEngine
        from opacus.validators import ModuleValidator
    except ImportError as exc:
        raise ImportError(
            "Opacus is required for DP-SGD. Install with `pip install opacus` "
            "or run scripts/setup_env.sh."
        ) from exc

    # Replace any modules that aren't differentiable per-sample (BatchNorm etc.)
    if not ModuleValidator.is_valid(model):
        model = ModuleValidator.fix(model)

    privacy_engine = PrivacyEngine()
    private_model, private_optimizer, private_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=dataloader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        poisson_sampling=poisson_sampling,
    )

    return PrivateClientHandles(
        model=private_model,
        optimizer=private_optimizer,
        dataloader=private_loader,
        privacy_engine=privacy_engine,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )


def get_epsilon(
    handles: PrivateClientHandles,
    delta: float = 1e-5,
) -> float:
    """Return the cumulative $(\\varepsilon, \\delta)$-DP $\\varepsilon$ spent so far.

    Args:
        handles: per-client handles returned by :func:`make_private_client`.
        delta: DP failure probability $\\delta$ (default $10^{-5}$).

    Returns:
        Current $\\varepsilon$ from Opacus's RDP accountant.
    """
    return float(handles.privacy_engine.get_epsilon(delta=delta))
