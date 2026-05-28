"""Barabási–Albert preferential-attachment topology generator.

The BA model produces a power-law degree distribution: new nodes attach
preferentially to existing high-degree nodes, yielding a small number of
high-degree hubs and a long tail of low-degree leaves. This is the
canonical model for organically grown peer-to-peer networks, and a
useful stress test for topology-aware leverage allocation because the
resulting degree heterogeneity is extreme (high-degree hubs in a
50-node BA graph at m=2 can have degree >20 while many leaves have
degree exactly m=2).

We use ``networkx``'s reference implementation (already a project
dependency) and convert the result into Murmura-compatible
:class:`~fulcrum.topology.Topology` form.

Reference:
    Barabási, A.-L. and Albert, R. (1999). "Emergence of scaling in
    random networks." *Science* 286 (5439): 509–512.
    https://doi.org/10.1126/science.286.5439.509
"""

from __future__ import annotations

from fulcrum.topology import Topology


def make_barabasi_albert(num_clients: int, m: int = 2, seed: int = 12345) -> Topology:
    """Generate a Barabási–Albert preferential-attachment graph.

    Args:
        num_clients: number of nodes $n$. Must be at least 2.
        m: number of edges each new node attaches to existing nodes.
            Each new node connects to $m$ existing nodes with probability
            proportional to their current degree. Must satisfy
            $1 \\leq m < n$. The expected mean degree of the resulting
            graph is approximately $2m$ (asymptotically).
        seed: random seed for the topology generator. Fixed across
            experiment.seed values so that the topology itself is a
            deployment constant; only the partitioning and training
            trajectory vary across replicates.

    Returns:
        :class:`~fulcrum.topology.Topology` with the adjacency list and
        edge list of the generated graph.

    Raises:
        ValueError: if ``num_clients`` < 2 or ``m`` is out of range.
    """
    if num_clients < 2:
        raise ValueError(f"num_clients must be >= 2 for BA, got {num_clients}")
    if not (1 <= m < num_clients):
        raise ValueError(
            f"m must be in [1, {num_clients - 1}] (got {m}); "
            "BA attaches each new node to m existing nodes."
        )

    # Local import keeps `networkx` out of the module-import hot path
    # (Topology imports `fulcrum.topology` which imports Murmura's
    # Topology; we want barabasi_albert to be opt-in via the dispatcher
    # in runner.py rather than eagerly imported at package load).
    import networkx as nx

    G = nx.barabasi_albert_graph(num_clients, m, seed=seed)
    neighbors: list[list[int]] = [sorted(G.neighbors(i)) for i in range(num_clients)]
    edges: list[tuple[int, int]] = [(int(u), int(v)) for u, v in G.edges()]
    return Topology(num_nodes=num_clients, neighbors=neighbors, edges=edges)
