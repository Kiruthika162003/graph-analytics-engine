from __future__ import annotations

import pytest

from mesh.errors import Invalid, Missing
from mesh.graph import Graph


def _triangle(directed: bool = False) -> Graph:
    g = Graph(directed=directed)
    for n in "abc":
        g.add_node(n)
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    g.add_edge("c", "a")
    return g


class TestNodes:
    def test_adding_a_node_makes_it_present(self):
        g = Graph()
        g.add_node("x")
        assert g.has_node("x")

    def test_adding_a_node_twice_is_idempotent(self):
        g = Graph()
        g.add_node("x")
        g.add_node("x")
        assert g.node_count() == 1


class TestEdges:
    def test_an_undirected_edge_is_symmetric(self):
        g = _triangle()
        assert g.has_edge("a", "b")
        assert g.has_edge("b", "a")

    def test_a_directed_edge_is_one_way(self):
        g = _triangle(directed=True)
        assert g.has_edge("a", "b")
        assert not g.has_edge("b", "a")

    def test_an_edge_to_a_missing_node_is_refused(self):
        g = Graph()
        g.add_node("a")
        with pytest.raises(Missing):
            g.add_edge("a", "ghost")

    def test_the_edge_count_ignores_the_mirror_direction(self):
        g = _triangle()
        assert g.edge_count() == 3

    def test_a_weight_is_stored_and_returned(self):
        g = Graph()
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b", weight=2.5)
        assert g.weight("a", "b") == 2.5

    def test_a_missing_edge_weight_is_refused(self):
        g = _triangle()
        with pytest.raises(Missing):
            g.weight("a", "z")


class TestDegree:
    def test_degree_counts_neighbors(self):
        g = _triangle()
        assert g.degree("a") == 2

    def test_in_and_out_degree_split_in_a_directed_graph(self):
        g = _triangle(directed=True)
        assert g.out_degree("a") == 1
        assert g.in_degree("a") == 1

    def test_handshake_lemma_holds(self):
        g = _triangle()
        assert sum(g.degree(n) for n in g.nodes()) == 2 * g.edge_count()


class TestReverse:
    def test_reverse_flips_every_directed_edge(self):
        g = _triangle(directed=True)
        r = g.reverse()
        assert r.has_edge("b", "a")
        assert not r.has_edge("a", "b")

    def test_reversing_an_undirected_graph_is_refused(self):
        with pytest.raises(Invalid):
            _triangle().reverse()
