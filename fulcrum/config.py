"""Experiment configuration schema.

YAML-loadable, dataclass-based. The schema mirrors the sections you'd see in a
config file:

.. code-block:: yaml

    experiment:
      name: setting_a_canonical
      seed: 0
      mode: target          # target | shadow
      output_dir: runs

    setting: A              # A | B | C
    data:
      data_root: data
      # Setting C only:
      n_clients: 50
      eta: 0.5
      alpha: 0.5
      sensitive_class: 0

    topology:
      type: hierarchical    # hierarchical | line | ring | complete | erdos | k-regular
      num_regions: 3        # hierarchical only
      num_nodes: 50         # ignored if data.n_clients is set
      # k for k-regular, p for erdos, etc.
      params: {}

    training:
      rounds: 100
      local_epochs: 1
      batch_size: 32
      learning_rate: 0.01

    dp:
      enabled: true
      allocation: topology_aware  # topology_aware | uniform | none
      utility_budget_U: 0.5
      observation_window: 100
      max_grad_norm: 1.0
      delta: 1.0e-5

    leverage:
      proxy: group_size     # group_size | degree | eta_position | uniform

    attack:
      enabled: true
      channels: [A1, A2_topo, A2_org, A2_full]
      regressor: lightgbm

The runner consumes :class:`ExperimentConfig` directly. CLI tooling loads YAML
into this schema via :func:`load_config`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class ExperimentMeta:
    name: str = "unnamed"
    seed: int = 0
    mode: Literal["target", "shadow"] = "target"
    output_dir: str = "runs"


@dataclass
class DataConfig:
    data_root: str = "data"
    # Setting C only — ignored for A/B
    n_clients: int | None = None
    eta: float | None = None
    alpha: float | None = None
    sensitive_class: int | None = None


@dataclass
class TopologyConfig:
    type: Literal[
        "hierarchical", "line", "ring", "complete", "erdos", "k-regular", "fully", "star"
    ] = "hierarchical"
    num_regions: int | None = None
    num_nodes: int | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    rounds: int = 100
    local_epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 0.01


@dataclass
class DPConfig:
    enabled: bool = True
    allocation: Literal["topology_aware", "uniform", "none"] = "topology_aware"
    utility_budget_U: float = 0.5
    observation_window: int = 100
    max_grad_norm: float = 1.0
    delta: float = 1.0e-5


@dataclass
class LeverageConfig:
    proxy: Literal["group_size", "degree", "eta_position", "uniform"] = "group_size"


@dataclass
class AttackConfig:
    enabled: bool = True
    channels: list[str] = field(default_factory=lambda: ["A1", "A2_topo", "A2_org", "A2_full"])
    regressor: Literal["lightgbm", "mlp", "linear"] = "lightgbm"


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Root configuration for one experiment."""

    setting: Literal["A", "B", "C"]
    experiment: ExperimentMeta = field(default_factory=ExperimentMeta)
    data: DataConfig = field(default_factory=DataConfig)
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dp: DPConfig = field(default_factory=DPConfig)
    leverage: LeverageConfig = field(default_factory=LeverageConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)

    def hash_id(self) -> str:
        """Stable SHA-256 hash over the full config — the run identifier.

        Two configs with identical fields produce identical IDs, so re-running
        a completed experiment is a no-op for the factorial manager.
        """
        data = asdict(self)
        # Sort keys recursively for stability
        s = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(s).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)


# ---------------------------------------------------------------------------
# YAML I/O
# ---------------------------------------------------------------------------

def _from_dict(cls, data: dict[str, Any]):
    """Recursively instantiate dataclass ``cls`` from a dict, preserving defaults."""
    if data is None:
        return cls()
    init_kwargs = {}
    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    for name, raw in data.items():
        if name not in field_types:
            raise ValueError(f"Unknown field {name!r} for {cls.__name__}")
        # Nested dataclass support: only descend if raw is a dict and field is dataclass-like
        if isinstance(raw, dict):
            sub_cls = _resolve_subdataclass(cls, name)
            if sub_cls is not None:
                init_kwargs[name] = _from_dict(sub_cls, raw)
                continue
        init_kwargs[name] = raw
    return cls(**init_kwargs)


def _resolve_subdataclass(parent_cls, field_name: str):
    """Best-effort lookup of a nested dataclass type by field name."""
    # Map of (parent, field) → sub-dataclass
    table = {
        (ExperimentConfig, "experiment"): ExperimentMeta,
        (ExperimentConfig, "data"): DataConfig,
        (ExperimentConfig, "topology"): TopologyConfig,
        (ExperimentConfig, "training"): TrainingConfig,
        (ExperimentConfig, "dp"): DPConfig,
        (ExperimentConfig, "leverage"): LeverageConfig,
        (ExperimentConfig, "attack"): AttackConfig,
    }
    return table.get((parent_cls, field_name))


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an :class:`ExperimentConfig` from a YAML file."""
    path = Path(path)
    with path.open("r") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping, got {type(raw).__name__}")
    if "setting" not in raw:
        raise ValueError("Config must specify 'setting' (one of A, B, C)")
    return _from_dict(ExperimentConfig, raw)


def dump_config(cfg: ExperimentConfig, path: str | Path) -> None:
    """Write an :class:`ExperimentConfig` to a YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        fh.write(cfg.to_yaml())
