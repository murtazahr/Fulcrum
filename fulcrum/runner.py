"""End-to-end FL training runner with topology-aware DP allocation + TADI feature collection.

This module is the integration glue: it takes an :class:`~fulcrum.config.ExperimentConfig`,
builds the data + topology + leverage + allocation, runs FedAvg with per-client
DP-SGD over $T_{\\max}$ rounds, collects per-round per-client features, and writes
all artifacts to disk + the SQLite experiments DB.

The training loop is intentionally simple FedAvg (or hierarchical FedAvg for
Setting A) with full participation. Subsampling is a follow-up extension.

Usage::

    from fulcrum.config import load_config
    from fulcrum.runner import run_experiment

    cfg = load_config("configs/setting_a_canonical.yaml")
    result = run_experiment(cfg, db_path="experiments.db")
    print(result["test_accuracy"], result["K_star"])
"""

from __future__ import annotations

import copy
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from fulcrum.config import ExperimentConfig
from fulcrum import storage as st


# ---------------------------------------------------------------------------
# Setting → (clients, topology, omega) builders
# ---------------------------------------------------------------------------

def _build_clients_topology(cfg: ExperimentConfig):
    """Construct per-setting clients, topology, omega.

    Returns:
        ``(clients, topology, omega)`` where ``clients`` is a list of
        :class:`~fulcrum.data.ClientDataset`, ``topology`` is a Murmura-compatible
        :class:`~fulcrum.topology.Topology`, and ``omega`` is a length-$n$
        organizational-label vector.
    """
    setting = cfg.setting

    if setting == "A":
        from fulcrum.data.flamby_isic import FedISICConfig, default_organizational_labels, make_client_datasets
        from fulcrum.topology.hierarchical import make_hierarchical
        sensitive = cfg.data.sensitive_class if cfg.data.sensitive_class is not None else 0
        clients = make_client_datasets(FedISICConfig(sensitive_class=sensitive))
        num_regions = cfg.topology.num_regions if cfg.topology.num_regions is not None else 3
        topology, omega_default = make_hierarchical(num_clients=len(clients), num_regions=num_regions)
        # Prefer the canonical Setting A omega when n=6 and num_regions=3
        if len(clients) == 6 and num_regions == 3:
            omega = default_organizational_labels()
        else:
            omega = omega_default
        return clients, topology, omega

    if setting == "B":
        from fulcrum.data.flamby_heart import FedHeartConfig, make_client_datasets
        sensitive = cfg.data.sensitive_class if cfg.data.sensitive_class is not None else 1
        clients = make_client_datasets(FedHeartConfig(sensitive_class=sensitive))
        topology = _build_topology(cfg.topology, len(clients))
        omega = np.arange(len(clients), dtype=np.int64)  # each client = own group
        return clients, topology, omega

    if setting == "C":
        from fulcrum.data.cifar10_eta import PartitionConfig, make_client_datasets
        if cfg.data.n_clients is None or cfg.data.eta is None or cfg.data.alpha is None:
            raise ValueError("Setting C requires data.{n_clients, eta, alpha} to be set")
        sensitive = cfg.data.sensitive_class if cfg.data.sensitive_class is not None else 0
        pcfg = PartitionConfig(
            n_clients=cfg.data.n_clients,
            eta=cfg.data.eta,
            alpha=cfg.data.alpha,
            sensitive_class=sensitive,
            seed=cfg.experiment.seed,
        )
        clients = make_client_datasets(pcfg, cfg.data.data_root)
        topology = _build_topology(cfg.topology, len(clients))
        omega = np.arange(len(clients), dtype=np.int64)  # each client own group for Setting C
        return clients, topology, omega

    raise ValueError(f"Unknown setting: {setting!r}")


def _build_topology(tcfg, num_nodes: int):
    """Dispatch to the correct topology generator based on type."""
    type_ = tcfg.type
    n = tcfg.num_nodes if tcfg.num_nodes is not None else num_nodes

    if type_ == "hierarchical":
        from fulcrum.topology.hierarchical import make_hierarchical
        num_regions = tcfg.num_regions or 3
        topo, _omega = make_hierarchical(num_clients=n, num_regions=num_regions)
        return topo
    if type_ == "line":
        from fulcrum.topology.line import make_line
        return make_line(n)

    # Murmura's built-in topologies (ring, fully/complete, erdos, k-regular, star)
    try:
        from murmura.topology.generators import create_topology
    except ImportError as exc:
        raise ImportError(
            f"Topology type {type_!r} requires Murmura. Install via scripts/setup_env.sh."
        ) from exc
    name = "fully" if type_ in ("complete", "fully") else type_
    if name == "star":
        # Star isn't in Murmura's set; emulate with fully-connected for the simulator
        # and document the difference. (Star matters for hierarchy interpretation, not for
        # peer-graph DP analysis under our threat model.)
        name = "fully"
    return create_topology(name, num_nodes=n, **tcfg.params)


# ---------------------------------------------------------------------------
# Leverage → allocation
# ---------------------------------------------------------------------------

def _build_allocation(cfg: ExperimentConfig, topology, omega: np.ndarray):
    """Compute the per-client noise allocation under the chosen leverage proxy."""
    from fulcrum.dp.allocation import optimal_allocation, uniform_allocation
    from fulcrum.dp.leverage import (
        leverage_degree, leverage_eta_position, leverage_group_size, leverage_uniform,
    )

    n = topology.num_nodes
    proxy = cfg.leverage.proxy
    if proxy == "group_size":
        leverage = leverage_group_size(omega)
    elif proxy == "degree":
        leverage = leverage_degree(topology.neighbors)
    elif proxy == "eta_position":
        eta = cfg.data.eta if cfg.data.eta is not None else 0.0
        leverage = leverage_eta_position(n, eta=eta, n_classes=10)
    elif proxy == "uniform":
        leverage = leverage_uniform(n)
    else:
        raise ValueError(f"Unknown leverage proxy: {proxy!r}")

    # a = T_max * C^2 / (2 |B|^2)
    a = (cfg.dp.observation_window * cfg.dp.max_grad_norm ** 2) / (2 * cfg.training.batch_size ** 2)

    if cfg.dp.allocation == "topology_aware":
        alloc = optimal_allocation(leverage, a=a, U=cfg.dp.utility_budget_U)
    elif cfg.dp.allocation == "uniform":
        alloc = uniform_allocation(n, U=cfg.dp.utility_budget_U, leverage=leverage, a=a)
    elif cfg.dp.allocation == "none":
        # Sentinel: no DP. We still return an Allocation with very high noise=0 to express "no noise".
        from fulcrum.dp.allocation import Allocation
        alloc = Allocation(
            sigma_squared=np.zeros(n),
            K_star=float("inf"),
            K_uniform=float("inf"),
            budget=0.0,
            is_uniform=True,
        )
    else:
        raise ValueError(f"Unknown allocation: {cfg.dp.allocation!r}")
    return leverage, alloc, a


# ---------------------------------------------------------------------------
# FL training loop with per-client DP-SGD + feature collection
# ---------------------------------------------------------------------------

def _train_one_round(model, dataloader, optimizer, criterion, device) -> float:
    """Train ``model`` for one local epoch over ``dataloader``. Returns mean loss."""
    import torch
    model.train()
    total = 0.0
    count = 0
    for batch in dataloader:
        if isinstance(batch, (list, tuple)):
            inputs, targets = batch[0], batch[1]
        else:
            inputs, targets = batch["input"], batch["target"]
        inputs = inputs.to(device)
        targets = targets.to(device)
        if targets.dtype == torch.float32 and targets.ndim > 1:
            # FLamby Heart Disease returns float labels; squeeze + cast to long
            targets = targets.squeeze(-1).long()
        elif targets.dtype != torch.long:
            targets = targets.long()
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * inputs.size(0)
        count += inputs.size(0)
    return total / max(count, 1)


def _evaluate(model, dataloader, criterion, device) -> tuple[float, float]:
    """Return (mean_loss, accuracy) on the given dataloader."""
    import torch
    model.eval()
    total_loss = 0.0
    correct = 0
    count = 0
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch[0], batch[1]
            else:
                inputs, targets = batch["input"], batch["target"]
            inputs = inputs.to(device)
            targets = targets.to(device)
            if targets.dtype == torch.float32 and targets.ndim > 1:
                targets = targets.squeeze(-1).long()
            elif targets.dtype != torch.long:
                targets = targets.long()
            logits = model(inputs)
            loss = criterion(logits, targets)
            preds = logits.argmax(dim=-1)
            total_loss += float(loss.item()) * inputs.size(0)
            correct += int((preds == targets).sum().item())
            count += inputs.size(0)
    return total_loss / max(count, 1), correct / max(count, 1)


def _fedavg_aggregate(state_dicts: list[dict], weights: np.ndarray) -> dict:
    """Weighted average of state dicts (FedAvg). All state dicts must share keys."""
    import torch
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    keys = state_dicts[0].keys()
    out = {}
    for k in keys:
        stacked = torch.stack([sd[k].detach().to(torch.float32) for sd in state_dicts], dim=0)
        w = torch.tensor(weights, dtype=torch.float32, device=stacked.device).view(-1, *([1] * (stacked.dim() - 1)))
        out[k] = (stacked * w).sum(dim=0)
    return out


def _hierarchical_aggregate(state_dicts: list[dict], omega: np.ndarray, weights: np.ndarray) -> dict:
    """Two-level FedAvg: average within each region (using ``weights``), then average across regions."""
    n_regions = int(omega.max()) + 1
    region_states = []
    region_weights = []
    for r in range(n_regions):
        members = [i for i, om in enumerate(omega) if int(om) == r]
        if not members:
            continue
        sub_states = [state_dicts[i] for i in members]
        sub_weights = weights[members]
        region_states.append(_fedavg_aggregate(sub_states, sub_weights))
        region_weights.append(float(sub_weights.sum()))
    return _fedavg_aggregate(region_states, np.asarray(region_weights))


def run_experiment(
    cfg: ExperimentConfig,
    db_path: str | Path = "experiments.db",
    runs_root: str | Path | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Execute one experiment end-to-end.

    Idempotent: if a run with this config's hash already has status ``done``,
    returns its summary metrics without re-running.

    Returns a dict with the headline metrics: ``run_id``, ``test_accuracy``,
    ``final_loss``, ``K_star``, ``K_uniform``, ``epsilon_max``.
    """
    import torch

    runs_root = Path(runs_root) if runs_root else Path(cfg.experiment.output_dir)
    runs_root.mkdir(parents=True, exist_ok=True)
    run_id = st.register_run(db_path, cfg, output_dir=runs_root)

    existing_status = st.get_run_status(db_path, run_id)
    if existing_status == "done":
        return {"run_id": run_id, "status": "done", "skipped": True}

    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    st.write_config_json(run_dir, cfg)
    st.mark_started(db_path, run_id)

    try:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(cfg.experiment.seed)
        np.random.seed(cfg.experiment.seed)

        # 1. Build clients, topology, omega
        clients, topology, omega = _build_clients_topology(cfg)
        n = len(clients)
        weights = np.array([c.n_train for c in clients], dtype=np.float64)
        weights = weights / max(weights.sum(), 1.0)

        # 2. Compute leverage and allocation
        leverage, alloc, a_coef = _build_allocation(cfg, topology, omega)

        # 3. Build model and per-client optimizers + DP wrappers
        from fulcrum.models import make_model
        from torch.utils.data import DataLoader

        global_model = make_model(cfg.setting).to(device)
        criterion = torch.nn.CrossEntropyLoss()

        per_client_loaders = [
            DataLoader(c.train_dataset, batch_size=cfg.training.batch_size, shuffle=True)
            for c in clients
        ]
        per_client_test_loaders = [
            DataLoader(c.test_dataset, batch_size=cfg.training.batch_size)
            for c in clients
        ]

        # Wrap each client with Opacus DP-SGD if DP is enabled
        from fulcrum.dp.opacus_wrap import make_private_client
        dp_handles = [None] * n
        if cfg.dp.enabled and cfg.dp.allocation != "none":
            from fulcrum.models import make_model as _make_model
            for i in range(n):
                local_model = _make_model(cfg.setting).to(device)
                local_model.load_state_dict(global_model.state_dict())
                opt = torch.optim.SGD(local_model.parameters(), lr=cfg.training.learning_rate)
                sigma_i = float(np.sqrt(alloc.sigma_squared[i]))
                dp_handles[i] = make_private_client(
                    local_model, opt, per_client_loaders[i],
                    noise_multiplier=max(sigma_i, 1e-6),
                    max_grad_norm=cfg.dp.max_grad_norm,
                    poisson_sampling=False,
                )

        # 4. Training loop with feature collection
        T_max = min(cfg.training.rounds, cfg.dp.observation_window)
        from fulcrum.attacks.features import collect_round_features
        feature_buffer: list[list[np.ndarray]] = [[] for _ in range(n)]

        for t in range(T_max):
            round_state_dicts = []
            for i in range(n):
                if dp_handles[i] is not None:
                    local_model = dp_handles[i].model
                    opt = dp_handles[i].optimizer
                    loader = dp_handles[i].dataloader
                else:
                    local_model = make_model(cfg.setting).to(device)
                    local_model.load_state_dict(global_model.state_dict())
                    opt = torch.optim.SGD(local_model.parameters(), lr=cfg.training.learning_rate)
                    loader = per_client_loaders[i]
                # Sync local from global at the start of each round
                local_model.load_state_dict(global_model.state_dict())
                # Local training
                for _ in range(cfg.training.local_epochs):
                    _train_one_round(local_model, loader, opt, criterion, device)
                # Collect features
                feat = collect_round_features(
                    local_model.state_dict(),
                    global_state_dict=global_model.state_dict(),
                )
                feature_buffer[i].append(feat)
                round_state_dicts.append({k: v.detach().cpu() for k, v in local_model.state_dict().items()})
            # Aggregate
            if cfg.setting == "A" and cfg.topology.type == "hierarchical":
                new_state = _hierarchical_aggregate(round_state_dicts, omega, weights)
            else:
                new_state = _fedavg_aggregate(round_state_dicts, weights)
            global_model.load_state_dict({k: v.to(device) for k, v in new_state.items()})

        # 5. Evaluate global model
        all_losses = []
        all_accs = []
        for i in range(n):
            loss, acc = _evaluate(global_model, per_client_test_loaders[i], criterion, device)
            all_losses.append(loss)
            all_accs.append(acc)
        test_loss = float(np.mean(all_losses))
        test_acc = float(np.mean(all_accs))

        # 6. Build the raw feature buffer in [n_clients, T_max, F_raw] layout
        # Pad shorter sequences (in case of failures) — they should all be T_max
        max_t = max(len(buf) for buf in feature_buffer)
        f_dim = feature_buffer[0][0].size if feature_buffer[0] else 0
        raw = np.zeros((n, max_t, f_dim), dtype=np.float32)
        for i, buf in enumerate(feature_buffer):
            for t, feat in enumerate(buf):
                raw[i, t, :feat.size] = feat

        # 7. Ground truth p_i
        p_true = np.array([c.p_sensitive for c in clients], dtype=np.float64)

        # 8. ε accounting (optional — only meaningful if DP enabled)
        epsilon_vals = []
        if dp_handles[0] is not None:
            from fulcrum.dp.opacus_wrap import get_epsilon
            for h in dp_handles:
                if h is not None:
                    try:
                        epsilon_vals.append(get_epsilon(h, delta=cfg.dp.delta))
                    except Exception:
                        pass
        epsilon_max = float(max(epsilon_vals)) if epsilon_vals else None

        # 9. Persist artifacts
        feat_path = st.write_features_npz(run_dir, raw, p_true, topology.neighbors, omega=omega)
        metrics = {
            "run_id": run_id,
            "test_accuracy": test_acc,
            "test_loss": test_loss,
            "K_star": alloc.K_star if np.isfinite(alloc.K_star) else None,
            "K_uniform": alloc.K_uniform if np.isfinite(alloc.K_uniform) else None,
            "epsilon_max": epsilon_max,
            "leverage": leverage.tolist(),
            "sigma_squared": alloc.sigma_squared.tolist(),
            "p_true": p_true.tolist(),
            "n_clients": n,
            "T_max": T_max,
            "wall_clock_seconds": time.time(),
        }
        metrics_path = st.write_metrics_json(run_dir, metrics)

        st.mark_done(
            db_path, run_id,
            test_accuracy=test_acc, final_loss=test_loss,
            K_star=alloc.K_star if np.isfinite(alloc.K_star) else None,
            K_uniform=alloc.K_uniform if np.isfinite(alloc.K_uniform) else None,
            epsilon=epsilon_max,
            config_path=run_dir / "config.json",
            features_path=feat_path,
            metrics_path=metrics_path,
        )
        return {"run_id": run_id, "status": "done", **metrics}

    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        st.mark_failed(db_path, run_id, err)
        raise
