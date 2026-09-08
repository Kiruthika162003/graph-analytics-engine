from __future__ import annotations

import random
from itertools import combinations

import pytest

from mesh.errors import Invalid, Missing
from mesh.graph import Graph
from mesh.triangles import Triangles


def _complete(k: int) -> Graph:
    g = Graph()
    nodes = [str(i) for i in range(k)]
    for n in nodes:
        g.add_node(n)
    for a, b in combinations(nodes, 2):
        g.add_edge(a, b)
    return g


def _brute_triangles(g: Graph) -> int:
    return sum(
        1
        for a, b, c in combinations(g.nodes(), 3)
        if g.has_edge(a, b) and g.has_edge(b, c) and g.has_edge(a, c)
    )


class TestCount:
    def test_a_single_triangle_counts_once(self):
        assert Triangles(_complete(3)).total == 1

    def test_k4_has_four_triangles(self):
        assert Triangles(_complete(4)).total == 4

    def test_a_star_has_no_triangles(self):
        g = Graph()
        g.add_node("hub")
        for leaf in "abcd":
            g.add_node(leaf)
            g.add_edge("hub", leaf)
        assert Triangles(g).total == 0

    def test_each_corner_of_a_triangle_sees_it_once(self):
        t = Triangles(_complete(3))
        assert all(t.per_node[n] == 1 for n in "012")


class TestClustering:
    def test_a_complete_graph_clusters_fully(self):
        t = Triangles(_complete(4))
        assert t.clustering("0") == 1.0
        assert t.average_clustering() == 1.0
        assert t.transitivity() == pytest.approx(1.0)

    def test_a_star_hub_clusters_at_zero(self):
        g = Graph()
        g.add_node("hub")
        for leaf in "abc":
            g.add_node(leaf)
            g.add_edge("hub", leaf)
        assert Triangles(g).clustering("hub") == 0.0

    def test_a_degree_one_node_clusters_at_zero(self):
        g = _complete(3)
        g.add_node("tail")
        g.add_edge("0", "tail")
        assert Triangles(g).clustering("tail") == 0.0

    def test_a_half_connected_neighborhood_clusters_at_half(self):
        # hub with neighbors a,b,c; only a-b is connected: 1 of 3 pairs
        g = Graph()
        for n in ["hub", "a", "b", "c"]:
            g.add_node(n)
        for leaf in "abc":
            g.add_edge("hub", leaf)
        g.add_edge("a", "b")
        assert Triangles(g).clustering("hub") == pytest.approx(1 / 3)


class TestRefusals:
    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            Triangles(Graph(directed=True))

    def test_a_missing_node_is_refused(self):
        with pytest.raises(Missing):
            Triangles(_complete(3)).clustering("ghost")


class TestAgainstBruteForce:
    def test_the_count_matches_checking_every_triple(self):
        rng = random.Random(83)
        for _ in range(30):
            g = Graph()
            nodes = [str(i) for i in range(9)]
            for n in nodes:
                g.add_node(n)
            for a, b in combinations(nodes, 2):
                if rng.random() < 0.4:
                    g.add_edge(a, b)
            assert Triangles(g).total == _brute_triangles(g)


class TestReport:
    def test_the_note_states_the_triangle_count(self):
        assert "4 triangle(s)" in Triangles(_complete(4)).note()
