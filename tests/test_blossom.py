from __future__ import annotations

import random
from itertools import combinations

import pytest

from mesh.blossom import Blossom
from mesh.errors import Invalid
from mesh.graph import Graph


def _brute_max_matching(g: Graph) -> int:
    edges = [(u, v) for u, v, _w in g.edges()]
    for k in range(len(edges), 0, -1):
        for subset in combinations(edges, k):
            seen: set[str] = set()
            ok = True
            for u, v in subset:
                if u in seen or v in seen:
                    ok = False
                    break
                seen.update((u, v))
            if ok:
                return k
    return 0


def _cycle(k: int) -> Graph:
    g = Graph()
    nodes = [str(i) for i in range(k)]
    for n in nodes:
        g.add_node(n)
    for i in range(k):
        g.add_edge(nodes[i], nodes[(i + 1) % k])
    return g


class TestMatching:
    def test_an_odd_cycle_matches_all_but_one(self):
        b = Blossom(_cycle(5))
        assert b.size() == 2
        assert b.is_valid()

    def test_an_even_cycle_is_perfectly_matched(self):
        b = Blossom(_cycle(6))
        assert b.size() == 3

    def test_the_triangle_with_tails_needs_a_blossom(self):
        # two triangles joined by a path: a plain alternating search stalls
        g = Graph()
        for n in "abcdef":
            g.add_node(n)
        for u, v in [("a", "b"), ("b", "c"), ("c", "a"), ("c", "d"),
                     ("d", "e"), ("e", "f"), ("f", "d")]:
            g.add_edge(u, v)
        b = Blossom(g)
        assert b.size() == 3
        assert b.is_valid()

    def test_a_star_matches_exactly_one_edge(self):
        g = Graph()
        g.add_node("hub")
        for leaf in "abc":
            g.add_node(leaf)
            g.add_edge("hub", leaf)
        assert Blossom(g).size() == 1

    def test_the_matching_uses_real_edges_and_no_node_twice(self):
        g = _cycle(7)
        g.add_edge("0", "3")
        assert Blossom(g).is_valid()


class TestRefusal:
    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            Blossom(Graph(directed=True))


class TestAgainstBruteForce:
    def test_the_size_matches_exhaustive_search_on_random_graphs(self):
        rng = random.Random(367)
        for _ in range(40):
            g = Graph()
            nodes = [str(i) for i in range(8)]
            for n in nodes:
                g.add_node(n)
            for a, c in combinations(nodes, 2):
                if rng.random() < 0.35:
                    g.add_edge(a, c)
            b = Blossom(g)
            assert b.is_valid()
            assert b.size() == _brute_max_matching(g)


class TestReport:
    def test_the_note_counts_contractions(self):
        assert "blossom contraction(s)" in Blossom(_cycle(5)).note()
