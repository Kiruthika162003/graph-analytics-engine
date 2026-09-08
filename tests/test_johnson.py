from __future__ import annotations

import random

import pytest

from mesh.errors import Negative, Unreachable
from mesh.floydwarshall import FloydWarshall
from mesh.graph import Graph
from mesh.johnson import Johnson


def _negative_edge_graph() -> Graph:
    g = Graph(directed=True)
    for n in "abcd":
        g.add_node(n)
    g.add_edge("a", "b", 4)
    g.add_edge("a", "c", 1)
    g.add_edge("c", "b", -2)
    g.add_edge("b", "d", 3)
    return g


class TestDistances:
    def test_a_negative_edge_shortens_the_route(self):
        # a->c->b is 1 + (-2) = -1, beating the direct a->b of 4
        assert Johnson(_negative_edge_graph()).distance("a", "b") == -1

    def test_reweighted_edges_are_all_nonnegative(self):
        j = Johnson(_negative_edge_graph())
        h = j.potential
        for u, v, w in _negative_edge_graph().edges():
            assert w + h[u] - h[v] >= 0

    def test_an_unreachable_pair_is_refused(self):
        g = _negative_edge_graph()
        g.add_node("island")
        with pytest.raises(Unreachable):
            Johnson(g).distance("a", "island")

    def test_a_negative_cycle_is_refused(self):
        g = Graph(directed=True)
        for n in "abc":
            g.add_node(n)
        g.add_edge("a", "b", 1)
        g.add_edge("b", "c", -3)
        g.add_edge("c", "a", 1)
        with pytest.raises(Negative):
            Johnson(g)


class TestAgainstFloydWarshall:
    def test_johnson_matches_floyd_warshall_on_random_graphs(self):
        rng = random.Random(41)
        for _ in range(30):
            g = Graph(directed=True)
            nodes = [str(i) for i in range(7)]
            for n in nodes:
                g.add_node(n)
            for u in nodes:
                for v in nodes:
                    if u != v and rng.random() < 0.3:
                        g.add_edge(u, v, rng.randint(-2, 9))
            try:
                fw = FloydWarshall(g)
            except Negative:
                with pytest.raises(Negative):
                    Johnson(g)
                continue
            j = Johnson(g)
            for u in nodes:
                for v in nodes:
                    assert j.dist[u][v] == pytest.approx(fw.dist[u][v])


class TestReport:
    def test_spread_is_zero_with_no_negative_edges(self):
        g = Graph(directed=True)
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b", 3)
        assert Johnson(g).potential_spread() == 0

    def test_spread_is_positive_with_a_negative_edge(self):
        assert Johnson(_negative_edge_graph()).potential_spread() > 0

    def test_the_note_names_the_spread(self):
        assert "potential spread" in Johnson(_negative_edge_graph()).note()
