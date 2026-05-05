"""Tests for fulcrum.topology.line and fulcrum.topology.hierarchical.

Properties verified:
- Line: degree sequence (1, 2, 2, ..., 2, 1) and connectivity.
- Hierarchical: omega assignment is contiguous; within-region complete subgraphs;
  region-size invariants under uneven n / num_regions.
- Both: edges list and neighbors list are mutually consistent (every
  ``(a, b) ∈ edges`` implies ``b ∈ neighbors[a]`` and vice versa).
"""

from __future__ import annotations

import numpy as np
import pytest

from fulcrum.topology import Topology
from fulcrum.topology.hierarchical import make_hierarchical, make_hierarchical_setting_a
from fulcrum.topology.line import make_line


def _consistency_check(topo: Topology) -> None:
    """Helper: every edge is reflected in both endpoints' neighbor lists, no duplicates."""
    edge_set = {tuple(sorted(e)) for e in topo.edges}
    for a in range(topo.num_nodes):
        for b in topo.neighbors[a]:
            assert tuple(sorted((a, b))) in edge_set, f"edge ({a},{b}) missing from edges list"
            assert a in topo.neighbors[b], f"asymmetric: {a} ∈ N({b}) missing"
    # No duplicate neighbors
    for a, nbrs in enumerate(topo.neighbors):
        assert len(nbrs) == len(set(nbrs)), f"duplicate neighbors at node {a}: {nbrs}"


# ---------------------------------------------------------------------------
# Line topology
# ---------------------------------------------------------------------------

class TestLineTopology:
    def test_degenerate_single_node(self):
        topo = make_line(1)
        assert topo.num_nodes == 1
        assert topo.neighbors == [[]]
        assert topo.edges == []

    def test_two_nodes(self):
        topo = make_line(2)
        assert topo.num_nodes == 2
        assert topo.neighbors == [[1], [0]]
        assert topo.edges == [(0, 1)]
        _consistency_check(topo)

    def test_degree_sequence(self):
        # Line of 5 nodes: degrees should be (1, 2, 2, 2, 1)
        topo = make_line(5)
        degrees = [len(nbrs) for nbrs in topo.neighbors]
        assert degrees == [1, 2, 2, 2, 1]
        _consistency_check(topo)

    def test_edge_count(self):
        # Line of n has n-1 edges
        for n in [3, 5, 10, 50]:
            topo = make_line(n)
            assert len(topo.edges) == n - 1

    def test_invalid_size(self):
        with pytest.raises(ValueError, match="num_nodes must be"):
            make_line(0)
        with pytest.raises(ValueError, match="num_nodes must be"):
            make_line(-1)


# ---------------------------------------------------------------------------
# Hierarchical topology
# ---------------------------------------------------------------------------

class TestHierarchicalTopology:
    def test_setting_a_canonical(self):
        # 6 sites, 3 regions of 2
        topo, omega = make_hierarchical_setting_a()
        assert topo.num_nodes == 6
        np.testing.assert_array_equal(omega, [0, 0, 1, 1, 2, 2])
        # Each region of size 2 -> one edge per region -> 3 edges total
        assert len(topo.edges) == 3
        _consistency_check(topo)

    def test_uneven_assignment(self):
        # 7 clients, 3 regions -> sizes (3, 2, 2)
        topo, omega = make_hierarchical(num_clients=7, num_regions=3)
        np.testing.assert_array_equal(omega, [0, 0, 0, 1, 1, 2, 2])
        # Region 0 (3 clients) -> C(3,2) = 3 edges
        # Region 1 (2 clients) -> 1 edge
        # Region 2 (2 clients) -> 1 edge
        # Total -> 5 edges
        assert len(topo.edges) == 5
        _consistency_check(topo)

    def test_within_region_complete_subgraphs(self):
        # 4 clients, 2 regions of 2 -> client 0-1 connected, client 2-3 connected
        topo, _omega = make_hierarchical(num_clients=4, num_regions=2)
        assert sorted(topo.neighbors[0]) == [1]
        assert sorted(topo.neighbors[1]) == [0]
        assert sorted(topo.neighbors[2]) == [3]
        assert sorted(topo.neighbors[3]) == [2]

    def test_single_region_is_complete_graph(self):
        # All clients in one region -> complete graph (n choose 2 edges)
        topo, omega = make_hierarchical(num_clients=4, num_regions=1)
        assert len(topo.edges) == 6  # C(4,2)
        np.testing.assert_array_equal(omega, [0, 0, 0, 0])

    def test_singleton_regions_no_edges(self):
        # n regions with n clients -> each region has 1 client -> no edges
        topo, omega = make_hierarchical(num_clients=5, num_regions=5)
        assert topo.edges == []
        np.testing.assert_array_equal(omega, [0, 1, 2, 3, 4])

    def test_more_regions_than_clients_raises(self):
        with pytest.raises(ValueError, match="must be >= num_regions"):
            make_hierarchical(num_clients=2, num_regions=3)

    def test_invalid_size(self):
        with pytest.raises(ValueError, match="must be >= 1"):
            make_hierarchical(num_clients=0, num_regions=1)
        with pytest.raises(ValueError, match="must be >= 1"):
            make_hierarchical(num_clients=3, num_regions=0)
