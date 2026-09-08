from __future__ import annotations

import pytest

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
    def test_the_tree_has_node_count_minus_one_edges(self):
        k = Kruskal(_weighted())
        assert len(k.edges()) == _weighted().node_count() - 1

    def test_the_total_weight_is_minimal(self):
        # a-b(1), then a-c or b-c(2), then c-d(3): total 6, avoiding a-d(10)
        assert Kruskal(_weighted()).total_weight() == 6

    def test_the_tree_spans_a_connected_graph(self):
        assert Kruskal(_weighted()).spans()

    def test_every_chosen_edge_is_a_real_edge(self):
        g = _weighted()
        for u, v, _w in Kruskal(g).edges():
            assert g.has_edge(u, v)


class TestForest:
    def test_a_disconnected_graph_yields_a_forest(self):
        g = Graph()
        for n in "abcd":
            g.add_node(n)
        g.add_edge("a", "b", 1)
        g.add_edge("c", "d", 1)  # two separate pairs
        k = Kruskal(g)
        assert not k.spans()
        assert k.tree_count() == 2


class TestRefusalAndReport:
    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            Kruskal(Graph(directed=True))

    def test_the_note_states_the_total_weight(self):
        assert "total weight 6" in Kruskal(_weighted()).note()
