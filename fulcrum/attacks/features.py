"""Feature extraction for TADI.

Two layers:

1. **Raw collection during training** (:func:`collect_round_features`) — called
   once per round per client. Computes layer-wise norms + cosine similarity to
   the global model and appends to the run's feature buffer. Cheap (~50 floats
   per (client, round)), keeps storage bounded regardless of model size.

2. **Feature construction at attack time** (:func:`build_attack_features`) —
   takes the raw per-round feature buffer + topology + organizational labels
   and produces the per-client feature vector $x_i$ that the regressor sees.

The two-layer split is what keeps the per-run storage to ~10 KB even for
ResNet-18: we never persist full parameter tensors, only norm summaries.

Design ref: Redesign/03_attack_design.md §3.2
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch


# ---------------------------------------------------------------------------
# Layer 1 — collect per-round features during training
# ---------------------------------------------------------------------------

def collect_round_features(
    local_state_dict: "dict[str, torch.Tensor]",
    global_state_dict: "dict[str, torch.Tensor] | None" = None,
) -> np.ndarray:
    """Extract a layer-norm summary vector from a single client's model state.

    Output layout (length $L + L + 1$):
        [layer_norms (L), cosine_to_global_per_layer (L), full_norm (1)]
    where $L$ is the number of named tensors in ``local_state_dict``.

    Args:
        local_state_dict: client $i$'s model state at the end of round $t$.
        global_state_dict: server-aggregated state at end of round $t-1$ (optional).
            If None, cosine entries are filled with zeros.

    Returns:
        1D numpy array of float32. Stored verbatim per (client, round) in features.npz.
    """
    import torch  # local import keeps the module importable without torch installed
    layer_norms = []
    cos_per_layer = []
    sq = 0.0
    for name, tensor in local_state_dict.items():
        flat = tensor.detach().to(torch.float32).flatten()
        n = float(torch.linalg.vector_norm(flat).item())
        layer_norms.append(n)
        sq += n * n
        if global_state_dict is not None and name in global_state_dict:
            g = global_state_dict[name].detach().to(torch.float32).flatten()
            gn = float(torch.linalg.vector_norm(g).item())
            if n > 0 and gn > 0:
                cos_per_layer.append(float(torch.dot(flat, g).item()) / (n * gn))
            else:
                cos_per_layer.append(0.0)
        else:
            cos_per_layer.append(0.0)
    full_norm = float(np.sqrt(sq))
    return np.concatenate(
        [np.asarray(layer_norms, dtype=np.float32),
         np.asarray(cos_per_layer, dtype=np.float32),
         np.asarray([full_norm], dtype=np.float32)]
    )


# ---------------------------------------------------------------------------
# Layer 2 — build attack-time per-client feature vectors
# ---------------------------------------------------------------------------

def _temporal_aggregate(seq: np.ndarray) -> np.ndarray:
    """Reduce a per-round time series ``seq`` of shape (T, F) to a fixed-length vector.

    Per-feature aggregates: mean, std, last-round value, linear-trend slope.
    Returns a length-(4F) vector.
    """
    if seq.shape[0] < 2:
        # Trend slope undefined for length<2; pad with zeros.
        zeros = np.zeros(seq.shape[1], dtype=np.float32)
        if seq.shape[0] == 0:
            return np.concatenate([zeros, zeros, zeros, zeros])
        return np.concatenate([seq[0], zeros, seq[0], zeros])
    T = seq.shape[0]
    mean = seq.mean(axis=0)
    std = seq.std(axis=0)
    last = seq[-1]
    # Linear trend slope per feature: cov(t, x) / var(t)
    t = np.arange(T, dtype=np.float64) - (T - 1) / 2.0
    var_t = float((t * t).mean())
    cov = (t[:, None] * (seq.astype(np.float64) - mean[None, :])).mean(axis=0)
    slope = (cov / var_t).astype(np.float32) if var_t > 0 else np.zeros(seq.shape[1], dtype=np.float32)
    return np.concatenate([mean.astype(np.float32), std.astype(np.float32), last.astype(np.float32), slope])


def _structural_features(
    n_clients: int,
    neighbors: list[list[int]] | None,
    omega: np.ndarray | None,
    n_org_labels: int | None = None,
) -> np.ndarray:
    """Per-client structural features from $\\mathcal{G}$ and $\\omega$.

    Layout per client (when both inputs given):
        [degree (1), 1/degree (1), is_central (1), org_one_hot (n_org_labels)]
    Missing inputs are filled with zeros.

    Returns array of shape ``(n_clients, F_struct)``.
    """
    cols = []
    if neighbors is not None:
        deg = np.array([len(nbrs) for nbrs in neighbors], dtype=np.float32)
        inv_deg = np.where(deg > 0, 1.0 / np.maximum(deg, 1.0), 0.0).astype(np.float32)
        med = float(np.median(deg))
        is_central = (deg > med).astype(np.float32)
        cols += [deg.reshape(-1, 1), inv_deg.reshape(-1, 1), is_central.reshape(-1, 1)]
    if omega is not None:
        omega = np.asarray(omega, dtype=np.int64)
        if n_org_labels is None:
            n_org_labels = int(omega.max()) + 1
        one_hot = np.zeros((n_clients, n_org_labels), dtype=np.float32)
        one_hot[np.arange(n_clients), omega] = 1.0
        cols.append(one_hot)
    if not cols:
        return np.zeros((n_clients, 0), dtype=np.float32)
    return np.concatenate(cols, axis=1)


def build_attack_features(
    raw_features: np.ndarray,
    neighbors: list[list[int]] | None,
    omega: np.ndarray | None,
    channel: str = "A2_full",
    n_org_labels: int | None = None,
) -> np.ndarray:
    """Build the per-client feature matrix the regressor sees.

    Args:
        raw_features: array of shape ``(n_clients, T_max, F_raw)`` from
            :func:`collect_round_features` stacked across rounds.
        neighbors: Murmura-style adjacency list (or None to omit topology features).
        omega: organizational label vector of length $n$ (or None to omit).
        channel: ablation selector — one of ``"A1"`` (parameter-only),
            ``"A2_topo"`` (structural only), ``"A2_org"`` (organizational only),
            ``"A2_full"`` (combined).
        n_org_labels: total number of organizational groups (for one-hot width).
            Defaults to ``omega.max() + 1`` if omega is given.

    Returns:
        Array of shape ``(n_clients, F)`` where $F$ depends on the channel.

    Raises:
        ValueError: if the channel is unknown or required inputs are missing.
    """
    if channel not in {"A1", "A2_topo", "A2_org", "A2_full"}:
        raise ValueError(f"Unknown channel: {channel!r}")

    n = raw_features.shape[0]
    blocks = []

    # Parameter-side features (used by A1 and A2_full)
    if channel in {"A1", "A2_full"}:
        per_client_temporal = np.stack(
            [_temporal_aggregate(raw_features[i]) for i in range(n)],
            axis=0,
        )  # shape (n, 4 * F_raw)
        blocks.append(per_client_temporal)

    # Structural features (used by A2_topo, A2_org, A2_full)
    if channel in {"A2_topo", "A2_full"}:
        if neighbors is None:
            raise ValueError(f"channel={channel!r} requires neighbors but got None")
        struct_topo = _structural_features(n, neighbors, None)
        blocks.append(struct_topo)
    if channel in {"A2_org", "A2_full"}:
        if omega is None:
            raise ValueError(f"channel={channel!r} requires omega but got None")
        struct_org = _structural_features(n, None, omega, n_org_labels)
        blocks.append(struct_org)

    if not blocks:
        # A1 with empty raw features (n=0 case)
        return np.zeros((n, 0), dtype=np.float32)
    return np.concatenate(blocks, axis=1)
