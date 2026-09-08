from __future__ import annotations

import random
from itertools import combinations

import pytest

from mesh.errors import Invalid
from mesh.graph import Graph
from mesh.vertexcover import VertexCover


def _brute_min_cover(g: Graph) -> int:
    nodes = g.nodes()
    edges = g.edges()
    for size in range(0, len(nodes) + 1):
        for combo in combinations(nodes, size):
            chosen = set(combo)
            if all(u in chosen or v in chosen for u, v, _w in edges):
                return size
    return len(nodes)


def _random_bipartite(rng: random.Random) -> Graph:
    g = Graph()
    left = [f"l{i}" for i in range(4)]
    right = [f"r{i}" for i in range(4)]
    for n in left + right:
        g.add_node(n)
    for u in left:
        for v in right:
            if rng.random() < 0.4:
                g.add_edge(u, v)
    return g


class TestBipartiteExact:
    def test_a_star_is_covered_by_its_hub(self):
        g = Graph()
        g.add_node("hub")
        for leaf in "abc":
            g.add_node(leaf)
            g.add_edge("hub", leaf)
        vc = VertexCover(g)
        assert vc.exact
        assert vc.cover == {"hub"}

    def test_a_path_of_four_needs_two_guards(self):
        g = Graph()
        for n in "abcd":
            g.add_node(n)
        for u, v in [("a", "b"), ("b", "c"), ("c", "d")]:
            g.add_edge(u, v)
        vc = VertexCover(g)
        assert len(vc.cover) == 2
        assert vc.covers_every_edge()

    def test_the_exact_cover_matches_the_matching_size(self):
        rng = random.Random(17)
        for _ in range(30):
            g = _random_bipartite(rng)
            vc = VertexCover(g)
            assert vc.exact
            assert vc.covers_every_edge()
            assert len(vc.cover) == vc.lower_bound
            assert len(vc.cover) == _brute_min_cover(g)


class TestGeneralApprox:
    def test_a_triangle_is_not_exact_but_still_covers(self):
        g = Graph()
        for n in "abc":
            g.add_node(n)
        for u, v in combinations("abc", 2):
            g.add_edge(u, v)
        vc = VertexCover(g)
        assert not vc.exact
        assert vc.covers_every_edge()

    def test_the_approximation_stays_within_twice_the_optimum(self):
        rng = random.Random(19)
        for _ in range(30):
            g = Graph()
            nodes = [str(i) for i in range(7)]
            for n in nodes:
                g.add_node(n)
            for a, b in combinations(nodes, 2):
                if rng.random() < 0.4:
                    g.add_edge(a, b)
            vc = VertexCover(g)
            assert vc.covers_every_edge()
            optimum = _brute_min_cover(g)
            assert optimum <= len(vc.cover) <= 2 * optimum
            assert vc.lower_bound <= optimum


class TestRefusal:
    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            VertexCover(Graph(directed=True))


class TestReport:
    def test_the_note_names_the_regime(self):
        g = Graph()
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b")
        assert "exact by Konig" in VertexCover(g).note()
