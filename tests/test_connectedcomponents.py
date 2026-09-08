from __future__ import annotations

import pytest

from mesh.connectedcomponents import ConnectedComponents
from mesh.errors import Invalid, Missing
from mesh.graph import Graph


def _two_islands() -> Graph:
    # {a,b,c} connected, {d,e} connected, separate
    g = Graph()
    for n in "abcde":
        g.add_node(n)
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    g.add_edge("d", "e")
    return g


class TestComponents:
    def test_two_islands_are_two_components(self):
        assert ConnectedComponents(_two_islands()).count() == 2

    def test_the_component_of_a_node_is_its_island(self):
        cc = ConnectedComponents(_two_islands())
        assert cc.component_of("a") == {"a", "b", "c"}
        assert cc.component_of("d") == {"d", "e"}

    def test_components_are_listed_largest_first(self):
        comps = ConnectedComponents(_two_islands()).components()
        assert len(comps[0]) >= len(comps[1])

    def test_a_lone_node_is_its_own_component(self):
        g = _two_islands()
        g.add_node("lonely")
        cc = ConnectedComponents(g)
        assert cc.count() == 3
        assert cc.component_of("lonely") == {"lonely"}


class TestConnected:
    def test_a_single_island_is_connected(self):
        g = Graph()
        for n in "abc":
            g.add_node(n)
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        assert ConnectedComponents(g).is_connected()

    def test_two_islands_are_not_connected(self):
        assert not ConnectedComponents(_two_islands()).is_connected()


class TestGiant:
    def test_the_giant_share_is_the_largest_fraction(self):
        # 3 of 5 nodes in the biggest island
        assert ConnectedComponents(_two_islands()).giant_share() == 0.6

    def test_the_note_states_the_component_count(self):
        assert "2 component(s)" in ConnectedComponents(_two_islands()).note()


class TestRefusals:
    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            ConnectedComponents(Graph(directed=True))

    def test_a_missing_node_lookup_is_refused(self):
        with pytest.raises(Missing):
            ConnectedComponents(_two_islands()).component_of("ghost")
