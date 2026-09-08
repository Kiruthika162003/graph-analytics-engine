from __future__ import annotations

import random
from itertools import pairwise

import pytest

from mesh.errors import Invalid, Missing
from mesh.graph import Graph
from mesh.kruskal import Kruskal
from mesh.prim import Prim


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
    def test_the_tree_has_node_count_minus_one_edges(self):
        assert len(Prim(_weighted()).edges()) == 3

    def test_the_total_weight_is_minimal(self):
        assert Prim(_weighted()).total_weight() == 6

    def test_it_spans_a_connected_graph(self):
        assert Prim(_weighted()).spans()

    def test_it_does_not_span_a_disconnected_graph(self):
        g = Graph()
        for n in "abcd":
            g.add_node(n)
        g.add_edge("a", "b", 1)
        g.add_edge("c", "d", 1)
        assert not Prim(g, start="a").spans()


class TestRefusals:
    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            Prim(Graph(directed=True))

    def test_an_empty_graph_is_refused(self):
        with pytest.raises(Invalid):
            Prim(Graph())

    def test_a_missing_start_is_refused(self):
        with pytest.raises(Missing):
            Prim(_weighted(), start="ghost")


class TestAgreesWithKruskal:
    def test_prim_and_kruskal_reach_the_same_total_on_connected_graphs(self):
        rng = random.Random(202)
        for _ in range(40):
            g = Graph()
            nodes = [str(i) for i in range(8)]
            for n in nodes:
                g.add_node(n)
            # a random spanning path guarantees connectivity, plus extra edges
            shuffled = nodes[:]
            rng.shuffle(shuffled)
            for a, b in pairwise(shuffled):
                g.add_edge(a, b, rng.randint(1, 20))
            for _ in range(10):
                a, b = rng.sample(nodes, 2)
                if not g.has_edge(a, b):
                    g.add_edge(a, b, rng.randint(1, 20))
            assert Prim(g).total_weight() == Kruskal(g).total_weight()


class TestReport:
    def test_the_note_states_the_total_weight(self):
        assert "total weight 6" in Prim(_weighted()).note()
