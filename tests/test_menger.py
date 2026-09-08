from __future__ import annotations

import random
from itertools import combinations

import pytest

from mesh.bfs import BFS
from mesh.errors import Invalid, Missing
from mesh.graph import Graph
from mesh.menger import Menger


def _without_nodes(g: Graph, gone: set[str]) -> Graph:
    h = Graph(directed=g.directed)
    for n in g.nodes():
        if n not in gone:
            h.add_node(n)
    for u, v, w in g.edges():
        if u not in gone and v not in gone:
            h.add_edge(u, v, w)
    return h


def _min_node_separator(g: Graph, a: str, b: str) -> int:
    others = [n for n in g.nodes() if n not in (a, b)]
    for size in range(len(others) + 1):
        for gone in combinations(others, size):
            if not BFS(_without_nodes(g, set(gone)), a).reached(b):
                return size
    return len(others)


class TestCounts:
    def test_parallel_routes_count_separately(self):
        # a-x-b, a-y-b, a-z-b: three routes sharing only the endpoints
        g = Graph()
        for n in "abxyz":
            g.add_node(n)
        for mid in "xyz":
            g.add_edge("a", mid)
            g.add_edge(mid, "b")
        m = Menger(g, "a", "b")
        assert m.edge_disjoint == 3
        assert m.node_disjoint == 3

    def test_a_shared_hub_shows_as_the_gap(self):
        # a has two edges into hub h, h has two edges out to b via distinct
        # nodes: edge-disjoint 2, node-disjoint 1 because every route uses h
        g = Graph()
        for n in ["a", "p", "q", "h", "r", "s", "b"]:
            g.add_node(n)
        for u, v in [("a", "p"), ("a", "q"), ("p", "h"), ("q", "h"),
                     ("h", "r"), ("h", "s"), ("r", "b"), ("s", "b")]:
            g.add_edge(u, v)
        m = Menger(g, "a", "b")
        assert m.edge_disjoint == 2
        assert m.node_disjoint == 1
        assert m.shared_hub_gap() == 1

    def test_a_single_path_has_one_of_each(self):
        g = Graph()
        for n in "abc":
            g.add_node(n)
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        m = Menger(g, "a", "c")
        assert (m.edge_disjoint, m.node_disjoint) == (1, 1)

    def test_disconnected_endpoints_have_zero(self):
        g = Graph()
        g.add_node("a")
        g.add_node("b")
        assert Menger(g, "a", "b").edge_disjoint == 0

    def test_direction_matters_on_a_digraph(self):
        g = Graph(directed=True)
        for n in "abc":
            g.add_node(n)
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        assert Menger(g, "a", "c").edge_disjoint == 1
        assert Menger(g, "c", "a").edge_disjoint == 0


class TestRefusals:
    def test_equal_endpoints_are_refused(self):
        g = Graph()
        g.add_node("a")
        with pytest.raises(Invalid):
            Menger(g, "a", "a")

    def test_a_missing_endpoint_is_refused(self):
        g = Graph()
        g.add_node("a")
        with pytest.raises(Missing):
            Menger(g, "a", "ghost")


class TestAgainstSeparators:
    def test_node_disjoint_count_equals_the_smallest_node_separator(self):
        rng = random.Random(127)
        for _ in range(25):
            g = Graph()
            nodes = [str(i) for i in range(7)]
            for n in nodes:
                g.add_node(n)
            for x, y in combinations(nodes, 2):
                if rng.random() < 0.4:
                    g.add_edge(x, y)
            if g.has_edge("0", "6"):
                continue  # a direct edge cannot be separated by nodes
            m = Menger(g, "0", "6")
            assert m.node_disjoint == _min_node_separator(g, "0", "6")
            assert m.node_disjoint <= m.edge_disjoint


class TestReport:
    def test_the_note_states_both_counts(self):
        g = Graph()
        for n in "abxyz":
            g.add_node(n)
        for mid in "xyz":
            g.add_edge("a", mid)
            g.add_edge(mid, "b")
        assert "3 edge-disjoint and 3 node-disjoint" in Menger(g, "a", "b").note()
