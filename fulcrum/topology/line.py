"""Line topology generator.

A line graph $0$–$1$–$2$–$\\ldots$–$(n-1)$. Endpoints have degree 1, interior
nodes have degree 2. Used as a degenerate ablation in Setting C — the lowest
average degree among connected topologies.
"""

from __future__ import annotations

from fulcrum.topology import Topology


def make_line(num_nodes: int) -> Topology:
    """Create a line topology of $n$ nodes.

    Args:
        num_nodes: number of nodes $n$.

    Returns:
        :class:`Topology` with edges ``(0,1), (1,2), ..., (n-2, n-1)``.

    Raises:
        ValueError: if ``num_nodes < 1``.
    """
    if num_nodes < 1:
        raise ValueError(f"num_nodes must be >= 1, got {num_nodes}")

    if num_nodes == 1:
        return Topology(num_nodes=1, neighbors=[[]], edges=[])

    neighbors: list[list[int]] = [[] for _ in range(num_nodes)]
    edges: list[tuple[int, int]] = []
    for i in range(num_nodes - 1):
        j = i + 1
        neighbors[i].append(j)
        neighbors[j].append(i)
        edges.append((i, j))

    return Topology(num_nodes=num_nodes, neighbors=neighbors, edges=edges)
