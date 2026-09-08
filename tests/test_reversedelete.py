from __future__ import annotations

import random
from itertools import pairwise

import pytest

from mesh.errors import Invalid
from mesh.graph import Graph
from mesh.kruskal import Kruskal
from mesh.reversedelete import ReverseDelete


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
    def test_the_heaviest_cycle_edge_is_deleted_first(self):
        rd = ReverseDelete(_weighted())
        assert ("a", "d", 10) not in rd.kept

    def test_the_total_is_the_minimum(self):
        assert ReverseDelete(_weighted()).total_weight() == 6

    def test_it_spans_and_deletes_the_surplus(self):
        rd = ReverseDelete(_weighted())
        assert rd.spans()
        assert rd.deleted == 2

    def test_a_tree_deletes_nothing(self):
        g = Graph()
        for n in "abc":
            g.add_node(n)
        g.add_edge("a", "b", 1)
        g.add_edge("b", "c", 5)
        rd = ReverseDelete(g)
        assert rd.deleted == 0
        assert "already a tree" in rd.note()

    def test_a_disconnected_graph_keeps_a_forest(self):
        g = Graph()
        for n in "abcd":
            g.add_node(n)
        g.add_edge("a", "b", 1)
        g.add_edge("c", "d", 1)
        rd = ReverseDelete(g)
        assert not rd.spans()
        assert rd.deleted == 0


class TestRefusal:
    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            ReverseDelete(Graph(directed=True))


class TestAgreesWithKruskal:
    def test_reverse_delete_and_kruskal_reach_the_same_total(self):
        rng = random.Random(311)
        for _ in range(30):
            g = Graph()
            nodes = [str(i) for i in range(9)]
            for n in nodes:
                g.add_node(n)
            shuffled = nodes[:]
            rng.shuffle(shuffled)
            for a, b in pairwise(shuffled):
                g.add_edge(a, b, rng.randint(1, 15))
            for _ in range(12):
                a, b = rng.sample(nodes, 2)
                if not g.has_edge(a, b):
                    g.add_edge(a, b, rng.randint(1, 15))
            assert ReverseDelete(g).total_weight() == Kruskal(g).total_weight()


class TestReport:
    def test_the_note_counts_deletions(self):
        assert "deleted 2 of 5 edge(s)" in ReverseDelete(_weighted()).note()
