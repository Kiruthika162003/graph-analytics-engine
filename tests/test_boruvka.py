from __future__ import annotations

import random
from itertools import pairwise

import pytest

from mesh.boruvka import Boruvka
from mesh.errors import Invalid
from mesh.graph import Graph
from mesh.kruskal import Kruskal


def _weighted() -> Graph:
    g = Graph()
    for n in "abcd":
        g.add_node(n)
    g.add_edge("a", "b", 1)
    g.add_edge("b", "c", 2)
    g.add_edge("a", "c", 2)
    g.add_edge("c", "d", 3)
    g.add_edge("a", "d", 10)
    return g


class TestTree:
    def test_the_total_weight_is_minimal(self):
        assert Boruvka(_weighted()).total_weight() == 6

    def test_the_tree_has_node_count_minus_one_edges(self):
        assert len(Boruvka(_weighted()).edges()) == 3

    def test_it_spans_a_connected_graph(self):
        assert Boruvka(_weighted()).spans()

    def test_a_disconnected_graph_yields_a_forest(self):
        g = Graph()
        for n in "abcd":
            g.add_node(n)
        g.add_edge("a", "b", 1)
        g.add_edge("c", "d", 1)
        b = Boruvka(g)
        assert not b.spans()
        assert b.tree_count() == 2


class TestTies:
    def test_equal_weights_never_close_a_cycle(self):
        # a triangle of equal weights: naive per-component picks could take all 3
        g = Graph()
        for n in "abc":
            g.add_node(n)
        g.add_edge("a", "b", 1)
        g.add_edge("b", "c", 1)
        g.add_edge("c", "a", 1)
        b = Boruvka(g)
        assert len(b.edges()) == 2
        assert b.total_weight() == 2


class TestRounds:
    def test_rounds_stay_within_the_logarithmic_bound(self):
        rng = random.Random(53)
        for _ in range(20):
            g = Graph()
            nodes = [str(i) for i in range(16)]
            for n in nodes:
                g.add_node(n)
            shuffled = nodes[:]
            rng.shuffle(shuffled)
            for a, c in pairwise(shuffled):
                g.add_edge(a, c, rng.randint(1, 30))
            for _ in range(20):
                a, c = rng.sample(nodes, 2)
                if not g.has_edge(a, c):
                    g.add_edge(a, c, rng.randint(1, 30))
            b = Boruvka(g)
            assert b.rounds <= b.round_bound()


class TestRefusal:
    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            Boruvka(Graph(directed=True))


class TestAgreesWithKruskal:
    def test_boruvka_and_kruskal_reach_the_same_total(self):
        rng = random.Random(54)
        for _ in range(40):
            g = Graph()
            nodes = [str(i) for i in range(9)]
            for n in nodes:
                g.add_node(n)
            shuffled = nodes[:]
            rng.shuffle(shuffled)
            for a, c in pairwise(shuffled):
                g.add_edge(a, c, rng.randint(1, 8))
            for _ in range(12):
                a, c = rng.sample(nodes, 2)
                if not g.has_edge(a, c):
                    g.add_edge(a, c, rng.randint(1, 8))
            assert Boruvka(g).total_weight() == Kruskal(g).total_weight()


class TestReport:
    def test_the_note_states_rounds_and_weight(self):
        note = Boruvka(_weighted()).note()
        assert "round(s)" in note
        assert "total weight 6" in note
