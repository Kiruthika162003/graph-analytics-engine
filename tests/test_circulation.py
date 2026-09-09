from __future__ import annotations

import pytest

from mesh.circulation import BoundedFlow, Circulation
from mesh.errors import Invalid
from mesh.graph import Graph


def _ring(caps: dict[tuple[str, str], float]) -> Graph:
    g = Graph(directed=True)
    for n in "abcd":
        g.add_node(n)
    for (u, v), c in caps.items():
        g.add_edge(u, v, c)
    return g


class TestFeasibility:
    def test_a_ring_with_floors_below_every_ceiling_is_feasible(self):
        g = _ring({("a", "b"): 5, ("b", "c"): 5, ("c", "d"): 5, ("d", "a"): 5})
        c = Circulation(g, {("a", "b"): 2, ("c", "d"): 3})
        assert c.feasible
        assert c.within_bounds()
        assert c.conserves()
        # every arc of a ring must carry the same amount, so at least the largest floor
        assert all(f >= 3 for f in c.circulation.values())

    def test_a_floor_above_a_downstream_ceiling_is_infeasible(self):
        g = _ring({("a", "b"): 5, ("b", "c"): 2, ("c", "d"): 5, ("d", "a"): 5})
        c = Circulation(g, {("a", "b"): 4})
        assert not c.feasible
        assert c.shortfall() > 0
        assert "infeasible" in c.note()

    def test_no_floors_means_the_zero_circulation(self):
        g = _ring({("a", "b"): 5, ("b", "c"): 5, ("c", "d"): 5, ("d", "a"): 5})
        c = Circulation(g, {})
        assert c.feasible
        assert all(f == 0 for f in c.circulation.values())

    def test_a_dead_end_arc_with_a_floor_is_infeasible(self):
        g = Graph(directed=True)
        for n in "abc":
            g.add_node(n)
        g.add_edge("a", "b", 3)
        g.add_edge("b", "c", 3)
        c = Circulation(g, {("a", "b"): 1})
        assert not c.feasible

    def test_two_rings_sharing_a_node_balance_independently(self):
        g = Graph(directed=True)
        for n in "abcxy":
            g.add_node(n)
        for u, v in [("a", "b"), ("b", "c"), ("c", "a"), ("a", "x"), ("x", "y"), ("y", "a")]:
            g.add_edge(u, v, 4)
        c = Circulation(g, {("b", "c"): 2, ("x", "y"): 3})
        assert c.feasible
        assert c.conserves()
        assert c.circulation[("c", "a")] >= 2
        assert c.circulation[("y", "a")] >= 3


class TestBoundedFlow:
    def test_a_bounded_flow_reports_a_value_that_meets_the_floors(self):
        g = Graph(directed=True)
        for n in "sabt":
            g.add_node(n)
        g.add_edge("s", "a", 4)
        g.add_edge("s", "b", 4)
        g.add_edge("a", "t", 4)
        g.add_edge("b", "t", 4)
        bf = BoundedFlow(g, "s", "t", {("s", "a"): 2, ("b", "t"): 1})
        assert bf.feasible
        assert 3 <= bf.value <= 8
        assert "meets every floor" in bf.note()

    def test_a_floor_the_sink_cannot_absorb_fails(self):
        g = Graph(directed=True)
        for n in "sat":
            g.add_node(n)
        g.add_edge("s", "a", 5)
        g.add_edge("a", "t", 1)
        bf = BoundedFlow(g, "s", "t", {("s", "a"): 3})
        assert not bf.feasible
        assert bf.note() == "no flow meets every floor"


class TestRefusal:
    def test_bad_inputs_are_refused_by_name(self):
        g = _ring({("a", "b"): 5, ("b", "c"): 5, ("c", "d"): 5, ("d", "a"): 5})
        with pytest.raises(Invalid, match="no arc a->c"):
            Circulation(g, {("a", "c"): 1})
        with pytest.raises(Invalid, match=r"outside 0\.\.5"):
            Circulation(g, {("a", "b"): 9})
        with pytest.raises(Invalid):
            Circulation(Graph(), {})
        g.add_edge("b", "a", 1)
        with pytest.raises(Invalid, match="antiparallel"):
            Circulation(g, {})

    def test_a_direct_source_sink_arc_is_refused_for_bounded_flow(self):
        g = Graph(directed=True)
        g.add_node("s")
        g.add_node("t")
        g.add_edge("s", "t", 1)
        with pytest.raises(Invalid):
            BoundedFlow(g, "s", "t", {})


class TestReport:
    def test_the_note_states_the_total_moved(self):
        g = _ring({("a", "b"): 5, ("b", "c"): 5, ("c", "d"): 5, ("d", "a"): 5})
        note = Circulation(g, {("a", "b"): 2}).note()
        assert note.startswith("feasible circulation over 4 arc(s) moving")
