from __future__ import annotations

import random
from itertools import combinations

import pytest

from mesh.coloring import Coloring
from mesh.errors import Invalid
from mesh.graph import Graph


def _random_graph(rng: random.Random, n: int, p: float) -> Graph:
    g = Graph()
    nodes = [str(i) for i in range(n)]
    for node in nodes:
        g.add_node(node)
    for a, b in combinations(nodes, 2):
        if rng.random() < p:
            g.add_edge(a, b)
    return g


def _complete(k: int) -> Graph:
    g = Graph()
    nodes = [str(i) for i in range(k)]
    for n in nodes:
        g.add_node(n)
    for a, b in combinations(nodes, 2):
        g.add_edge(a, b)
    return g


class TestProper:
    @pytest.mark.parametrize("strategy", ["arbitrary", "welsh_powell", "dsatur"])
    def test_every_strategy_yields_a_proper_coloring(self, strategy: str):
        rng = random.Random(7)
        for _ in range(20):
            g = _random_graph(rng, 10, 0.4)
            c = Coloring(g, strategy=strategy)
            assert c.is_proper()

    @pytest.mark.parametrize("strategy", ["arbitrary", "welsh_powell", "dsatur"])
    def test_no_strategy_exceeds_max_degree_plus_one(self, strategy: str):
        rng = random.Random(8)
        for _ in range(20):
            g = _random_graph(rng, 10, 0.5)
            c = Coloring(g, strategy=strategy)
            assert c.color_count() <= c.upper_bound()


class TestOptimality:
    def test_a_complete_graph_needs_every_color(self):
        assert Coloring(_complete(5)).color_count() == 5

    def test_dsatur_two_colors_a_bipartite_graph(self):
        # an even cycle is bipartite; DSatur is exact there
        g = Graph()
        for n in "abcdef":
            g.add_node(n)
        for u, v in [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("e", "f"), ("f", "a")]:
            g.add_edge(u, v)
        assert Coloring(g, strategy="dsatur").color_count() == 2

    def test_an_odd_cycle_needs_three(self):
        g = Graph()
        for n in "abc":
            g.add_node(n)
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        g.add_edge("c", "a")
        assert Coloring(g).color_count() == 3

    def test_an_edgeless_graph_uses_one_color(self):
        g = Graph()
        for n in "abc":
            g.add_node(n)
        assert Coloring(g).color_count() == 1


class TestBounds:
    def test_the_lower_bound_is_three_when_a_triangle_exists(self):
        assert Coloring(_complete(4)).lower_bound() == 3

    def test_the_lower_bound_is_two_for_a_triangle_free_edge(self):
        g = Graph()
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b")
        assert Coloring(g).lower_bound() == 2

    def test_the_count_sits_between_the_bounds(self):
        rng = random.Random(9)
        for _ in range(20):
            c = Coloring(_random_graph(rng, 9, 0.4))
            assert c.lower_bound() <= c.color_count() <= c.upper_bound()


class TestRefusals:
    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            Coloring(Graph(directed=True))

    def test_an_unknown_strategy_is_refused(self):
        with pytest.raises(Invalid):
            Coloring(_complete(3), strategy="magic")


class TestReport:
    def test_the_note_states_the_count_and_bounds(self):
        note = Coloring(_complete(4)).note()
        assert "used 4 color(s)" in note
        assert "may not be optimal" in note
