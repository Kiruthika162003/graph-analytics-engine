from __future__ import annotations

from itertools import combinations

import pytest

from mesh.errors import Invalid
from mesh.graph import Graph
from mesh.richclub import RichClub


def _club_graph() -> Graph:
    # four hubs fully wired to each other, each with three private leaves
    g = Graph()
    hubs = ["h0", "h1", "h2", "h3"]
    for h in hubs:
        g.add_node(h)
    for a, b in combinations(hubs, 2):
        g.add_edge(a, b)
    for h in hubs:
        for i in range(3):
            leaf = f"{h}l{i}"
            g.add_node(leaf)
            g.add_edge(h, leaf)
    return g


class TestCoefficient:
    def test_the_hubs_form_a_complete_club(self):
        g = _club_graph()
        # hubs have degree 6, leaves degree 1: above 1 leaves only the hubs
        assert RichClub.coefficient(g, 1) == 1.0

    def test_a_level_with_fewer_than_two_rich_nodes_is_refused(self):
        with pytest.raises(Invalid):
            RichClub.coefficient(_club_graph(), 6)

    def test_the_baseline_keeps_every_degree(self):
        rc = RichClub(_club_graph(), seed=1)
        for n in rc.graph.nodes():
            assert rc.baseline.degree(n) == rc.graph.degree(n)
        assert rc.baseline.edge_count() == rc.graph.edge_count()

    def test_the_club_exceeds_its_degree_preserving_baseline(self):
        rc = RichClub(_club_graph(), seed=2)
        assert rc.normalized(1) >= 1.0

    def test_levels_stop_where_fewer_than_two_nodes_remain(self):
        rc = RichClub(_club_graph(), seed=3)
        assert rc.levels() == [0, 1, 2, 3, 4, 5]


class TestNoClub:
    def test_a_ring_has_no_rich_club_structure(self):
        g = Graph()
        nodes = [str(i) for i in range(12)]
        for n in nodes:
            g.add_node(n)
        for i in range(12):
            g.add_edge(nodes[i], nodes[(i + 1) % 12])
        rc = RichClub(g, seed=4)
        # every node has degree 2, so level 1 is everyone: raw equals baseline
        assert rc.normalized(1) == pytest.approx(1.0)


class TestRefusals:
    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            RichClub(Graph(directed=True))

    def test_too_few_edges_is_refused(self):
        g = Graph()
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b")
        with pytest.raises(Invalid):
            RichClub(g)


class TestReport:
    def test_the_note_names_a_verdict(self):
        note = RichClub(_club_graph(), seed=5).note()
        assert "normalized rich-club peaks" in note
