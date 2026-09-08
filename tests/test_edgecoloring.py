from __future__ import annotations

import random
from itertools import combinations

import pytest

from mesh.edgecoloring import EdgeColoring
from mesh.errors import Invalid
from mesh.graph import Graph


def _random_bipartite(rng: random.Random, n: int, p: float) -> Graph:
    g = Graph()
    left = [f"l{i}" for i in range(n)]
    right = [f"r{i}" for i in range(n)]
    for node in left + right:
        g.add_node(node)
    for u in left:
        for v in right:
            if rng.random() < p:
                g.add_edge(u, v)
    return g


class TestBipartiteExact:
    def test_a_complete_bipartite_graph_uses_exactly_its_degree(self):
        g = Graph()
        for n in ["a", "b", "c", "x", "y", "z"]:
            g.add_node(n)
        for u in "abc":
            for v in "xyz":
                g.add_edge(u, v)
        ec = EdgeColoring(g)
        assert ec.exact
        assert ec.is_proper()
        assert ec.color_count() == 3

    def test_an_even_cycle_needs_two(self):
        g = Graph()
        for n in "abcd":
            g.add_node(n)
        for u, v in [("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")]:
            g.add_edge(u, v)
        ec = EdgeColoring(g)
        assert ec.color_count() == 2

    def test_random_bipartite_graphs_always_land_on_the_max_degree(self):
        rng = random.Random(173)
        for _ in range(30):
            g = _random_bipartite(rng, 5, 0.5)
            if g.edge_count() == 0:
                continue
            ec = EdgeColoring(g)
            assert ec.is_proper()
            assert ec.color_count() == ec.max_degree


class TestGeneralGreedy:
    def test_a_triangle_needs_three_and_is_not_exact(self):
        g = Graph()
        for n in "abc":
            g.add_node(n)
        for u, v in combinations("abc", 2):
            g.add_edge(u, v)
        ec = EdgeColoring(g)
        assert not ec.exact
        assert ec.is_proper()
        assert ec.color_count() == 3

    def test_greedy_stays_proper_and_under_twice_the_degree(self):
        rng = random.Random(179)
        for _ in range(30):
            g = Graph()
            nodes = [str(i) for i in range(7)]
            for n in nodes:
                g.add_node(n)
            for a, b in combinations(nodes, 2):
                if rng.random() < 0.45:
                    g.add_edge(a, b)
            if g.edge_count() == 0:
                continue
            ec = EdgeColoring(g)
            assert ec.is_proper()
            assert ec.max_degree <= ec.color_count() <= 2 * ec.max_degree - 1


class TestRefusal:
    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            EdgeColoring(Graph(directed=True))


class TestReport:
    def test_the_note_names_the_regime_and_bounds(self):
        g = Graph()
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b")
        note = EdgeColoring(g).note()
        assert "exact by Konig" in note
        assert "Vizing's bound 2" in note
