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
    """Extract a per-client model-state summary vector at one round.

    Output layout (length $L + L + 1 + K + K$):
        [layer_norms        : L]   per-layer norm of local weights
        [cos_to_global      : L]   per-layer cosine to current global state
        [full_norm          : 1]   l2 norm across all layers
        [output_row_norms   : K]   l2 norm of each output-class row of the
                                   final classifier weight (signals which
                                   classes this client trained on)
        [output_delta_norms : K]   same, but on (local - global) for the
                                   classifier weight (per-class drift signal
                                   that survives DP noise better than the
                                   absolute weight magnitudes)

    where $L$ is the number of named tensors and $K$ is the output-class
    count (last linear layer's first axis).

    **Why the class-aware blocks** (added after v1 of the attack-eval gave
    negative attack lift on Setting B): the layer-norm / cosine features
    alone are *generic shape statistics* that don't distinguish "client saw
    class 3" from "client saw class 7". The final linear layer's per-row
    weights, however, are the model's output projection per class: a
    client with many class-c samples produces systematically larger
    gradient norms on row c of the classifier, which accumulates over
    rounds. The class-row delta vs global is the cleanest empirical
    surrogate for per-client class concentration and is the same signal
    Melis et al. (2019) exploited in the no-DP setting. DP noise reduces
    but does not eliminate it, because the per-class signal scales with
    the number of samples in that class while the noise scales with $C/|B|$
    independent of class.

    If the model has no obvious classifier layer (e.g., a sequential
    network with multiple linear layers), the heuristic picks the *last*
    linear-shaped parameter in the state dict (a 2D tensor) and treats
    its first dim as $K$. Setting $K = 2$ for Setting B's MLP, $K = 8$ for
    Setting A's CNN, $K = 10$ for Setting C's CNN.

    Args:
        local_state_dict: client $i$'s model state at the end of round $t$.
        global_state_dict: server-aggregated state at end of round $t-1$ (optional).
            If None, cosine and delta entries are filled with zeros.

    Returns:
        1D numpy array of float32. Stored verbatim per (client, round) in features.npz.
    """
    import torch  # local import keeps the module importable without torch installed
    layer_norms: list[float] = []
    cos_per_layer: list[float] = []
    sq = 0.0
    last_linear_local = None
    last_linear_global = None
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
        # Track the last 2D weight tensor — heuristic for the classifier.
        # Linear layers have shape (out_features, in_features); the first
        # dim is the class axis.
        if tensor.detach().dim() == 2:
            last_linear_local = tensor.detach().to(torch.float32)
            if global_state_dict is not None and name in global_state_dict:
                last_linear_global = global_state_dict[name].detach().to(torch.float32)
            else:
                last_linear_global = None

    full_norm = float(np.sqrt(sq))

    # Class-aware blocks. If no 2D weight tensor was found, emit empty
    # arrays — feature width then adapts per-run, and the attack-eval
    # join will still work for runs collected with the same model.
    if last_linear_local is not None and last_linear_local.dim() == 2:
        # Per-class row norms of the classifier weight
        per_class_norms = torch.linalg.vector_norm(last_linear_local, dim=1)
        output_row_norms = per_class_norms.cpu().numpy().astype(np.float32)
        if last_linear_global is not None and last_linear_global.shape == last_linear_local.shape:
            delta = last_linear_local - last_linear_global
            per_class_delta = torch.linalg.vector_norm(delta, dim=1)
            output_delta_norms = per_class_delta.cpu().numpy().astype(np.float32)
        else:
            output_delta_norms = np.zeros_like(output_row_norms)
    else:
        output_row_norms = np.zeros(0, dtype=np.float32)
        output_delta_norms = np.zeros(0, dtype=np.float32)

    return np.concatenate([
        np.asarray(layer_norms, dtype=np.float32),
        np.asarray(cos_per_layer, dtype=np.float32),
        np.asarray([full_norm], dtype=np.float32),
        output_row_norms,
        output_delta_norms,
    ])


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
