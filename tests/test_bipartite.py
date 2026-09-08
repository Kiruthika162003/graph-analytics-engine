from __future__ import annotations

import random

import pytest

from mesh.bipartite import Bipartite
from mesh.errors import Invalid
from mesh.graph import Graph


def _square() -> Graph:
    # an even cycle a-b-c-d-a is bipartite
    g = Graph()
    for n in "abcd":
        g.add_node(n)
    for u, v in [("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")]:
        g.add_edge(u, v)
    return g


def _triangle() -> Graph:
    g = Graph()
    for n in "abc":
        g.add_node(n)
    for u, v in [("a", "b"), ("b", "c"), ("c", "a")]:
        g.add_edge(u, v)
    return g


class TestVerdict:
    def test_an_even_cycle_is_bipartite(self):
        assert Bipartite(_square()).is_bipartite

    def test_a_triangle_is_not_bipartite(self):
        assert not Bipartite(_triangle()).is_bipartite

    def test_every_edge_crosses_the_sides_when_bipartite(self):
        g = _square()
        left, right = Bipartite(g).sides()
        for u, v, _w in g.edges():
            assert (u in left) != (v in left)
        assert left | right == set(g.nodes())

    def test_a_disconnected_graph_colors_each_component(self):
        g = _square()
        g.add_node("x")
        g.add_node("y")
        g.add_edge("x", "y")
        assert Bipartite(g).is_bipartite


class TestOddCycle:
    def test_the_odd_cycle_is_returned_with_odd_length(self):
        b = Bipartite(_triangle())
        assert len(b.odd_cycle) % 2 == 1

    def test_the_odd_cycle_walks_real_edges(self):
        g = _triangle()
        cycle = Bipartite(g).odd_cycle
        for i in range(len(cycle)):
            assert g.has_edge(cycle[i], cycle[(i + 1) % len(cycle)])

    def test_sides_are_refused_when_not_bipartite(self):
        with pytest.raises(Invalid):
            Bipartite(_triangle()).sides()


class TestRefusal:
    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            Bipartite(Graph(directed=True))


class TestAgainstBruteForce:
    def test_the_verdict_matches_trying_every_two_coloring(self):
        rng = random.Random(61)
        for _ in range(40):
            g = Graph()
            nodes = [str(i) for i in range(7)]
            for n in nodes:
                g.add_node(n)
            for u in nodes:
                for v in nodes:
                    if u < v and rng.random() < 0.3:
                        g.add_edge(u, v)
            edges = g.edges()
            exists = False
            for mask in range(1 << len(nodes)):
                side = {nodes[i] for i in range(len(nodes)) if mask >> i & 1}
                if all((u in side) != (v in side) for u, v, _w in edges):
                    exists = True
                    break
            assert Bipartite(g).is_bipartite == exists


class TestReport:
    def test_the_note_states_side_sizes(self):
        assert "sides of 2 and 2" in Bipartite(_square()).note()

    def test_the_note_names_the_odd_cycle_length(self):
        assert "odd cycle of length 3" in Bipartite(_triangle()).note()
