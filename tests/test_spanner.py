from __future__ import annotations

import random
from itertools import combinations

import pytest

from mesh.errors import Invalid
from mesh.graph import Graph
from mesh.kruskal import Kruskal
from mesh.spanner import GreedySpanner


def _random_weighted(rng: random.Random, n: int, p: float) -> Graph:
    g = Graph()
    nodes = [str(i) for i in range(n)]
    for node in nodes:
        g.add_node(node)
    for a, b in combinations(nodes, 2):
        if rng.random() < p:
            g.add_edge(a, b, rng.randint(1, 20))
    return g


def _complete(k: int) -> Graph:
    g = Graph()
    nodes = [str(i) for i in range(k)]
    for n in nodes:
        g.add_node(n)
    for a, b in combinations(nodes, 2):
        g.add_edge(a, b, 1)
    return g


class TestSpanner:
    def test_stretch_one_keeps_every_edge_of_a_unit_complete_graph(self):
        sp = GreedySpanner(_complete(5), stretch=1.0)
        assert sp.kept() == 10

    def test_a_large_stretch_thins_to_a_spanning_tree(self):
        g = _complete(6)
        sp = GreedySpanner(g, stretch=100.0)
        assert sp.kept() == g.node_count() - 1

    def test_the_stretch_bound_holds_on_every_pair(self):
        rng = random.Random(269)
        for _ in range(20):
            g = _random_weighted(rng, 9, 0.6)
            for t in (1.5, 2.0, 3.0):
                sp = GreedySpanner(g, stretch=t)
                assert sp.stretch_holds()

    def test_a_larger_stretch_never_keeps_more_edges(self):
        rng = random.Random(271)
        for _ in range(15):
            g = _random_weighted(rng, 9, 0.6)
            tight = GreedySpanner(g, stretch=1.5).kept()
            loose = GreedySpanner(g, stretch=3.0).kept()
            assert loose <= tight

    def test_the_spanner_contains_a_minimum_spanning_tree_weight_or_more(self):
        rng = random.Random(277)
        g = _random_weighted(rng, 10, 0.7)
        sp = GreedySpanner(g, stretch=2.0)
        assert Kruskal(sp.spanner).total_weight() == Kruskal(g).total_weight()


class TestRefusals:
    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            GreedySpanner(Graph(directed=True), stretch=2.0)

    def test_a_stretch_below_one_is_refused(self):
        with pytest.raises(Invalid):
            GreedySpanner(_complete(3), stretch=0.5)


class TestReport:
    def test_the_note_states_kept_and_observed(self):
        note = GreedySpanner(_complete(5), stretch=2.0).note()
        assert "kept" in note
        assert "worst observed" in note
