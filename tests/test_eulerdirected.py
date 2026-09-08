from __future__ import annotations

from collections import Counter
from itertools import pairwise

import pytest

from mesh.errors import Invalid
from mesh.eulerdirected import DirectedEulerian
from mesh.graph import Graph


def _uses_every_arrow_once(g: Graph, trail: list[str]) -> bool:
    walked = Counter(pairwise(trail))
    expected = Counter((u, v) for u, v, _w in g.edges())
    return walked == expected


def _two_cycles_sharing_a_node() -> Graph:
    # a->b->a and a->c->d->a: balanced, strongly connected, a circuit
    g = Graph(directed=True)
    for n in "abcd":
        g.add_node(n)
    for u, v in [("a", "b"), ("b", "a"), ("a", "c"), ("c", "d"), ("d", "a")]:
        g.add_edge(u, v)
    return g


class TestCircuit:
    def test_a_balanced_strongly_connected_graph_yields_a_circuit(self):
        e = DirectedEulerian(_two_cycles_sharing_a_node())
        assert e.is_circuit()
        assert e.trail[0] == e.trail[-1]
        assert _uses_every_arrow_once(_two_cycles_sharing_a_node(), e.trail)

    def test_every_step_follows_an_arrow_in_its_direction(self):
        g = _two_cycles_sharing_a_node()
        for u, v in pairwise(DirectedEulerian(g).trail):
            assert g.has_edge(u, v)


class TestOpenTrail:
    def test_one_surplus_out_and_one_surplus_in_give_an_open_trail(self):
        g = Graph(directed=True)
        for n in "abc":
            g.add_node(n)
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        e = DirectedEulerian(g)
        assert not e.is_circuit()
        assert e.trail == ["a", "b", "c"]
        assert e.imbalance_count() == 2

    def test_a_de_bruijn_style_overlap_graph_assembles(self):
        # k-mers as edges: AB->BC->CD with a loop BC->CB->BC
        g = Graph(directed=True)
        for n in ["AB", "BC", "CD", "CB"]:
            g.add_node(n)
        for u, v in [("AB", "BC"), ("BC", "CD"), ("BC", "CB"), ("CB", "BC")]:
            g.add_edge(u, v)
        e = DirectedEulerian(g)
        assert e.trail[0] == "AB" and e.trail[-1] == "CD"
        assert _uses_every_arrow_once(g, e.trail)


class TestRefusals:
    def test_an_unbalanced_node_is_named(self):
        g = Graph(directed=True)
        for n in "abc":
            g.add_node(n)
        g.add_edge("a", "b")
        g.add_edge("a", "c")  # a has two out, zero in
        with pytest.raises(Invalid) as caught:
            DirectedEulerian(g)
        assert "'a'" in str(caught.value)

    def test_balanced_but_not_strongly_connected_is_refused(self):
        g = Graph(directed=True)
        for n in "abcd":
            g.add_node(n)
        g.add_edge("a", "b")
        g.add_edge("b", "a")
        g.add_edge("c", "d")
        g.add_edge("d", "c")
        with pytest.raises(Invalid) as caught:
            DirectedEulerian(g)
        assert "strongly connected" in str(caught.value)

    def test_an_undirected_graph_is_refused(self):
        with pytest.raises(Invalid):
            DirectedEulerian(Graph())

    def test_an_edgeless_graph_is_refused(self):
        g = Graph(directed=True)
        g.add_node("a")
        with pytest.raises(Invalid):
            DirectedEulerian(g)


class TestReport:
    def test_the_note_names_the_kind_and_imbalance(self):
        note = DirectedEulerian(_two_cycles_sharing_a_node()).note()
        assert "directed Eulerian circuit" in note
        assert "0 unbalanced node(s)" in note
