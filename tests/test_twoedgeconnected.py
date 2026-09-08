from __future__ import annotations

import random
from itertools import combinations

import pytest

from mesh.connectedcomponents import ConnectedComponents
from mesh.errors import Invalid, Missing
from mesh.graph import Graph
from mesh.twoedgeconnected import TwoEdgeConnected


def _two_triangles_bridged() -> Graph:
    g = Graph()
    for n in "abcdef":
        g.add_node(n)
    for u, v in [("a", "b"), ("b", "c"), ("c", "a"),
                 ("d", "e"), ("e", "f"), ("f", "d"), ("c", "d")]:
        g.add_edge(u, v)
    return g


def _without_edge(g: Graph, a: str, b: str) -> Graph:
    h = Graph()
    for n in g.nodes():
        h.add_node(n)
    for u, v, w in g.edges():
        if {u, v} != {a, b}:
            h.add_edge(u, v, w)
    return h


class TestComponents:
    def test_the_bridge_splits_the_two_triangles(self):
        tec = TwoEdgeConnected(_two_triangles_bridged())
        assert len(tec.components) == 2
        assert tec.component_of("a") == {"a", "b", "c"}
        assert tec.component_of("e") == {"d", "e", "f"}

    def test_a_cycle_is_one_component_and_two_edge_connected(self):
        g = Graph()
        for n in "abcd":
            g.add_node(n)
        for u, v in [("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")]:
            g.add_edge(u, v)
        tec = TwoEdgeConnected(g)
        assert tec.is_two_edge_connected()
        assert tec.leaf_count() == 0

    def test_a_tree_is_all_singletons(self):
        g = Graph()
        for n in "abc":
            g.add_node(n)
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        tec = TwoEdgeConnected(g)
        assert len(tec.components) == 3
        assert not tec.is_two_edge_connected()


class TestBridgeTree:
    def test_the_bridge_tree_has_one_edge_per_bridge(self):
        tec = TwoEdgeConnected(_two_triangles_bridged())
        assert tec.bridge_tree.edge_count() == 1
        assert tec.leaf_count() == 2

    def test_one_edge_fixes_two_bridged_triangles(self):
        assert TwoEdgeConnected(_two_triangles_bridged()).edges_to_add() == 1

    def test_a_path_of_four_needs_two_edges(self):
        # bridge tree is a path of 4 nodes, 2 leaves, one edge pairs them
        g = Graph()
        for n in "abcd":
            g.add_node(n)
        for u, v in [("a", "b"), ("b", "c"), ("c", "d")]:
            g.add_edge(u, v)
        assert TwoEdgeConnected(g).edges_to_add() == 1


class TestRefusals:
    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            TwoEdgeConnected(Graph(directed=True))

    def test_a_missing_node_is_refused(self):
        with pytest.raises(Missing):
            TwoEdgeConnected(_two_triangles_bridged()).component_of("ghost")


class TestAgainstDefinition:
    def test_within_a_component_no_single_edge_removal_separates_a_pair(self):
        rng = random.Random(137)
        for _ in range(20):
            g = Graph()
            nodes = [str(i) for i in range(8)]
            for n in nodes:
                g.add_node(n)
            for a, b in combinations(nodes, 2):
                if rng.random() < 0.3:
                    g.add_edge(a, b)
            tec = TwoEdgeConnected(g)
            for comp in tec.components:
                members = sorted(comp)
                for x, y in combinations(members, 2):
                    for u, v, _w in g.edges():
                        reduced = _without_edge(g, u, v)
                        assert y in ConnectedComponents(reduced).component_of(x)


class TestReport:
    def test_the_note_counts_components_bridges_and_repairs(self):
        note = TwoEdgeConnected(_two_triangles_bridged()).note()
        assert "2 two-edge-connected component(s)" in note
        assert "1 bridge(s)" in note
        assert "1 edge(s) to add" in note
