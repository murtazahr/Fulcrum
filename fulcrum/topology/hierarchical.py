"""Hierarchical topology generator (Setting A).

Two-level FL hierarchy: clients → regional aggregators → global root. The
"topology" we expose to the simulator is the **within-region peer graph**
(clients in the same region are peers), and the inter-region structure is
encoded in the ``omega`` (organizational labels) vector.

This split lets us:

- Reuse Murmura's :class:`Topology` dataclass and its peer-graph training loop.
- Apply :func:`fulcrum.dp.leverage.leverage_group_size` directly to ``omega``.
- Implement hierarchical aggregation in the experiment runner as: average
  within each region, then average across regions to produce the global update.

Design ref: Redesign/02_threat_model_partitioning.md §2.6 Setting A
"""

from __future__ import annotations

import numpy as np

from fulcrum.topology import Topology


def make_hierarchical(
    num_clients: int,
    num_regions: int | None = None,
    region_sizes: list[int] | None = None,
    seed: int = 0,
) -> tuple[Topology, np.ndarray]:
    """Generate a hierarchical FL deployment.

    Within each region, clients form a complete subgraph (all peers); across
    regions there are no peer edges (the cross-region link goes through
    aggregators, not peer-to-peer).

    Two ways to specify the partition:

    - ``num_regions`` (only): clients distributed evenly. Resulting region_sizes
      are within ±1 of each other. **Produces uniform leverage** under
      degree- or group-size-based proxies, hence Theorem 2 collapses to uniform
      allocation. Useful as a degenerate-case control, not for headline figures.
    - ``region_sizes`` (explicit list): asymmetric region sizes. Required for
      non-trivial leverage variation. Sum must equal ``num_clients``.

    Args:
        num_clients: number of client nodes $n$.
        num_regions: number of regional aggregators (used if ``region_sizes`` is None).
        region_sizes: explicit per-region sizes (takes precedence if given).
        seed: random seed (currently unused — assignment is deterministic).

    Returns:
        Tuple ``(topology, omega)``:

        - ``topology``: peer graph with within-region complete subgraphs.
        - ``omega``: length-$n$ array assigning each client to its region.

    Raises:
        ValueError: invalid sizes or both/neither sizing arguments specified.
    """
    del seed  # currently deterministic; reserved for future randomized assignments
    if num_clients < 1:
        raise ValueError(f"num_clients must be >= 1, got {num_clients}")

    if region_sizes is not None:
        if num_regions is not None and num_regions != len(region_sizes):
            raise ValueError(
                f"num_regions={num_regions} contradicts len(region_sizes)={len(region_sizes)}"
            )
        if any(s < 1 for s in region_sizes):
            raise ValueError(f"region_sizes entries must be >= 1, got {region_sizes}")
        if sum(region_sizes) != num_clients:
            raise ValueError(
                f"sum(region_sizes)={sum(region_sizes)} != num_clients={num_clients}"
            )
        sizes = list(region_sizes)
    elif num_regions is not None:
        if num_regions < 1:
            raise ValueError(f"num_regions must be >= 1, got {num_regions}")
        if num_clients < num_regions:
            raise ValueError(
                f"num_clients ({num_clients}) must be >= num_regions ({num_regions})"
            )
        # Distribute clients across regions as evenly as possible.
        base, extra = divmod(num_clients, num_regions)
        sizes = [base + (1 if r < extra else 0) for r in range(num_regions)]
    else:
        raise ValueError("Must provide either num_regions or region_sizes")

    # omega[i] = region index for client i (assigned contiguously)
    omega = np.empty(num_clients, dtype=np.int64)
    cursor = 0
    region_member_lists: list[list[int]] = []
    for r, size in enumerate(sizes):
        members = list(range(cursor, cursor + size))
        omega[cursor:cursor + size] = r
        region_member_lists.append(members)
        cursor += size

    # Build within-region complete subgraphs
    neighbors: list[list[int]] = [[] for _ in range(num_clients)]
    edges: list[tuple[int, int]] = []
    for members in region_member_lists:
        for idx_a, a in enumerate(members):
            for b in members[idx_a + 1:]:
                neighbors[a].append(b)
                neighbors[b].append(a)
                edges.append((a, b))

    topology = Topology(num_nodes=num_clients, neighbors=neighbors, edges=edges)
    return topology, omega


def make_hierarchical_setting_a() -> tuple[Topology, np.ndarray]:
    """Convenience: the canonical Setting A topology (6 sites, 3 regions of 2)."""
    return make_hierarchical(num_clients=6, num_regions=3)
