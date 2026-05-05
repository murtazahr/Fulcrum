"""Topology generators not shipped by Murmura.

Murmura provides ring, fully, erdos, k-regular. Our experiments also need:

- :mod:`fulcrum.topology.line` — degenerate ablation; no closed cycle.
- :mod:`fulcrum.topology.hierarchical` — Setting A two-level hierarchy: clients
  → regional aggregators → root. Encoded as (within-region peer graph, omega).

Both generators return Murmura-compatible :class:`Topology` objects (or our
local fallback if Murmura is not installed).
"""

from __future__ import annotations

# Murmura-compatible Topology — try to use upstream, fall back to local definition.
try:
    from murmura.topology.base import Topology as _MurmuraTopology
    Topology = _MurmuraTopology
except ImportError:
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Topology:  # type: ignore[no-redef]
        """Local fallback compatible with murmura.topology.base.Topology.

        Identical field signature to Murmura's; we use this when Murmura is not
        importable (testing without the full environment).
        """

        num_nodes: int
        neighbors: list[list[int]]
        edges: list[tuple[int, int]]

        def degree(self, node_id: int) -> int:
            return len(self.neighbors[node_id])

        def avg_degree(self) -> float:
            if self.num_nodes == 0:
                return 0.0
            return sum(len(nbrs) for nbrs in self.neighbors) / self.num_nodes

        def is_connected(self) -> bool:
            if self.num_nodes == 0:
                return True
            visited = {0}
            queue = [0]
            while queue:
                node = queue.pop(0)
                for nbr in self.neighbors[node]:
                    if nbr not in visited:
                        visited.add(nbr)
                        queue.append(nbr)
            return len(visited) == self.num_nodes


__all__ = ["Topology"]
