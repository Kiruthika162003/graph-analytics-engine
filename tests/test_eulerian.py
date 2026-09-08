from __future__ import annotations

from collections import Counter
from itertools import pairwise

import pytest

from mesh.errors import Invalid
from mesh.eulerian import Eulerian
from mesh.graph import Graph


def _square() -> Graph:
    g = Graph()
    for n in "abcd":
        g.add_node(n)
    for u, v in [("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")]:
        g.add_edge(u, v)
    return g


def _uses_every_edge_once(g: Graph, trail: list[str]) -> bool:
    walked = Counter(frozenset(step) for step in pairwise(trail))
    expected = Counter(frozenset((u, v)) for u, v, _w in g.edges())
    return walked == expected


class TestCircuit:
    def test_an_even_graph_yields_a_closed_circuit(self):
        e = Eulerian(_square())
        assert e.is_circuit()
        assert e.trail[0] == e.trail[-1]

    def test_the_circuit_uses_every_edge_exactly_once(self):
        g = _square()
        assert _uses_every_edge_once(g, Eulerian(g).trail)

    def test_every_step_is_a_real_edge(self):
        g = _square()
        for u, v in pairwise(Eulerian(g).trail):
            assert g.has_edge(u, v)


class TestPath:
    def test_two_odd_nodes_yield_an_open_path_between_them(self):
        # a-b-c: a and c are odd, so the path runs from one to the other
        g = Graph()
        for n in "abc":
            g.add_node(n)
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        e = Eulerian(g)
        assert not e.is_circuit()
        assert {e.trail[0], e.trail[-1]} == {"a", "c"}

    def test_the_path_uses_every_edge_once_on_a_bridged_shape(self):
        # a square with a chain hanging off one corner: two odd nodes
        g = _square()
        g.add_node("e")
        g.add_edge("a", "e")
        assert _uses_every_edge_once(g, Eulerian(g).trail)


class TestRefusals:
    def test_more_than_two_odd_nodes_is_refused(self):
        # a star with three leaves has three odd nodes plus an odd hub
        g = Graph()
        g.add_node("hub")
        for leaf in "abc":
            g.add_node(leaf)
            g.add_edge("hub", leaf)
        with pytest.raises(Invalid) as caught:
            Eulerian(g)
        assert "odd degree" in str(caught.value)

    def test_edges_in_two_components_are_refused(self):
        g = Graph()
        for n in "abcd":
            g.add_node(n)
        g.add_edge("a", "b")
        g.add_edge("c", "d")
        with pytest.raises(Invalid):
            Eulerian(g)

    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            Eulerian(Graph(directed=True))

    def test_an_edgeless_graph_is_refused(self):
        g = Graph()
        g.add_node("a")
        with pytest.raises(Invalid):
            Eulerian(g)


class TestReport:
    def test_the_note_names_the_kind_and_odd_count(self):
        note = Eulerian(_square()).note()
        assert "Eulerian circuit" in note
        assert "0 odd-degree" in note
