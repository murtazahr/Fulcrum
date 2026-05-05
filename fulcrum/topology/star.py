"""Star topology generator.

One hub node connected to all leaves; leaves are not connected to each other.
The hub has degree $n-1$, leaves have degree $1$ — maximum degree asymmetry for
a connected $n$-node graph, hence the strongest leverage-variation case.

Used as the high-asymmetry anchor in Setting C topology sweeps. Concretely,
for $n=50$ at $\\eta=1$ the topology-aware advantage is ~1.2 nats — visibly
larger than line (~0.02 nats) or balanced hierarchical (~0 nats), making
star the right choice for fail-fast smoke runs that surface the gap quickly.
"""

from __future__ import annotations

from fulcrum.topology import Topology


def make_star(num_nodes: int, hub: int = 0) -> Topology:
    """Create a star topology with one hub and ``num_nodes - 1`` leaves.

    Args:
        num_nodes: total number of nodes (including the hub).
        hub: index of the hub node (default 0).

    Returns:
        :class:`Topology` with hub connected to every leaf and no leaf-leaf edges.

    Raises:
        ValueError: if ``num_nodes < 1`` or ``hub`` is out of range.
    """
    if num_nodes < 1:
        raise ValueError(f"num_nodes must be >= 1, got {num_nodes}")
    if not 0 <= hub < num_nodes:
        raise ValueError(f"hub must be in [0, {num_nodes}), got {hub}")

    if num_nodes == 1:
        return Topology(num_nodes=1, neighbors=[[]], edges=[])

    neighbors: list[list[int]] = [[] for _ in range(num_nodes)]
    edges: list[tuple[int, int]] = []
    for i in range(num_nodes):
        if i == hub:
            continue
        neighbors[hub].append(i)
        neighbors[i].append(hub)
        edges.append((min(i, hub), max(i, hub)))

    return Topology(num_nodes=num_nodes, neighbors=neighbors, edges=edges)
