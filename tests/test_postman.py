from __future__ import annotations

import pytest

from mesh.errors import Invalid
from mesh.graph import Graph
from mesh.postman import ChinesePostman


def _square() -> Graph:
    g = Graph()
    for n in "abcd":
        g.add_node(n)
    for u, v in [("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")]:
        g.add_edge(u, v, 1)
    return g


class TestEulerianCase:
    def test_all_even_degrees_cost_exactly_the_edge_total(self):
        pm = ChinesePostman(_square())
        assert pm.odd == []
        assert pm.extra == 0
        assert pm.route_cost() == 4


class TestOddNodes:
    def test_a_path_doubles_its_whole_length(self):
        # a-b-c: a and c are odd, the only pairing walks the path back
        g = Graph()
        for n in "abc":
            g.add_node(n)
        g.add_edge("a", "b", 2)
        g.add_edge("b", "c", 3)
        pm = ChinesePostman(g)
        assert pm.odd == ["a", "c"]
        assert pm.extra == 5
        assert pm.route_cost() == 10

    def test_a_pendant_on_a_cycle_doubles_only_the_pendant(self):
        g = _square()
        g.add_node("e")
        g.add_edge("a", "e", 7)
        pm = ChinesePostman(g)
        assert pm.odd == ["a", "e"]
        assert pm.extra == 7
        assert pm.pairing == [("a", "e")]

    def test_four_odd_nodes_pick_the_cheapest_of_three_pairings(self):
        # a star of four leaves: every pairing of leaves routes through the
        # hub, so all three pairings tie at 15 here and the point is that
        # the count of pairings considered is three and the extra is exact
        g = Graph()
        g.add_node("hub")
        for leaf, w in [("p", 1), ("q", 1), ("r", 10), ("s", 3)]:
            g.add_node(leaf)
            g.add_edge("hub", leaf, w)
        pm = ChinesePostman(g)
        assert len(pm.odd) == 4
        assert pm.pairings_considered() == 3
        # pairings: (p,q)+(r,s)=2+13=15; (p,r)+(q,s)=11+4=15; (p,s)+(q,r)=4+11=15
        assert pm.extra == 15


class TestRefusals:
    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            ChinesePostman(Graph(directed=True))

    def test_an_edgeless_graph_is_refused(self):
        g = Graph()
        g.add_node("a")
        with pytest.raises(Invalid):
            ChinesePostman(g)

    def test_a_disconnected_graph_is_refused(self):
        g = _square()
        g.add_node("x")
        g.add_node("y")
        g.add_edge("x", "y")
        with pytest.raises(Invalid):
            ChinesePostman(g)

    def test_too_many_odd_nodes_is_refused_with_the_cap_named(self):
        # a star with 12 leaves has 12 odd nodes
        g = Graph()
        g.add_node("hub")
        for i in range(12):
            g.add_node(f"l{i}")
            g.add_edge("hub", f"l{i}")
        with pytest.raises(Invalid) as caught:
            ChinesePostman(g)
        assert "blossom" in str(caught.value)


class TestReport:
    def test_the_note_states_base_and_extra(self):
        g = Graph()
        for n in "abc":
            g.add_node(n)
        g.add_edge("a", "b", 2)
        g.add_edge("b", "c", 3)
        note = ChinesePostman(g).note()
        assert "base 5 plus 5" in note
        assert "1 pairing(s)" in note
