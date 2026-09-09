from __future__ import annotations

import pytest

from mesh.errors import Invalid
from mesh.factories import cycle, path
from mesh.graph import Graph
from mesh.graphdiffreport import ChangeReport


def _copy(g: Graph) -> Graph:
    h = Graph(directed=g.directed)
    for n in g.nodes():
        h.add_node(n)
    for u, v, w in g.edges():
        h.add_edge(u, v, w)
    return h


class TestLevels:
    def test_a_chord_that_closes_a_triangle_adds_one_edge_and_one_triangle(self):
        before = cycle(5)
        after = _copy(before)
        after.add_edge("0", "2")
        cr = ChangeReport(before, after)
        assert cr.edges_added == [("0", "2")]
        assert cr.readings()["edges"] == (5.0, 6.0)
        assert cr.readings()["triangles"] == (0.0, 1.0)
        assert cr.degree_shifts() == [("0", 1), ("2", 1)]

    def test_removing_a_bridge_adds_a_piece(self):
        before = path(4)
        after = Graph()
        for n in before.nodes():
            after.add_node(n)
        after.add_edge("0", "1")
        after.add_edge("2", "3")
        cr = ChangeReport(before, after)
        assert cr.edges_removed == [("1", "2")]
        assert cr.readings()["pieces"] == (1.0, 2.0)
        assert "became less connected" in cr.note()
        assert "shrank" in cr.note()

    def test_new_nodes_and_their_edges_are_listed(self):
        before = path(2)
        after = _copy(before)
        after.add_node("hub")
        after.add_edge("hub", "0")
        after.add_edge("hub", "1")
        cr = ChangeReport(before, after)
        assert cr.nodes_added == ["hub"]
        assert cr.degree_shifts()[0] == ("hub", 2)
        assert "grew" in cr.note()

    def test_a_version_against_itself_reports_nothing(self):
        cr = ChangeReport(cycle(4), _copy(cycle(4)))
        assert cr.unchanged()
        assert cr.note() == "nothing changed at any level"


class TestDirected:
    def test_directed_keys_keep_their_orientation(self):
        before = Graph(directed=True)
        after = Graph(directed=True)
        for g in (before, after):
            g.add_node("a")
            g.add_node("b")
        before.add_edge("a", "b")
        after.add_edge("b", "a")
        cr = ChangeReport(before, after)
        assert cr.edges_added == [("b", "a")]
        assert cr.edges_removed == [("a", "b")]
        assert "triangles" not in cr.readings()

    def test_mixed_directions_are_refused(self):
        with pytest.raises(Invalid):
            ChangeReport(Graph(), Graph(directed=True))


class TestReport:
    def test_the_lines_carry_every_level(self):
        before = cycle(4)
        after = _copy(before)
        after.add_edge("0", "2")
        lines = ChangeReport(before, after).lines()
        assert lines[0] == "nodes +0 -0, edges +1 -0"
        assert any(line.startswith("edges: 4 to 5 (+1)") for line in lines)
        assert lines[-1] == "the graph grew and kept its pieces"
