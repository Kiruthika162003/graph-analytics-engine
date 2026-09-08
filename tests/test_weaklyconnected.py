from __future__ import annotations

import pytest

from mesh.errors import Invalid, Missing
from mesh.graph import Graph
from mesh.weaklyconnected import WeaklyConnected


def _one_way_chain() -> Graph:
    # a -> b -> c: one weak component, three strong ones
    g = Graph(directed=True)
    for n in "abc":
        g.add_node(n)
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    return g


class TestComponents:
    def test_a_one_way_chain_is_one_weak_component(self):
        wc = WeaklyConnected(_one_way_chain())
        assert wc.count() == 1
        assert wc.is_weakly_connected()

    def test_the_chain_has_three_strong_components(self):
        wc = WeaklyConnected(_one_way_chain())
        assert wc.strong_count() == 3
        assert wc.one_way_barriers() == 2

    def test_islands_are_separate_weak_components(self):
        g = _one_way_chain()
        g.add_node("x")
        g.add_node("y")
        g.add_edge("x", "y")
        wc = WeaklyConnected(g)
        assert wc.count() == 2
        assert wc.component_of("x") == {"x", "y"}

    def test_a_cycle_has_matching_weak_and_strong_counts(self):
        g = Graph(directed=True)
        for n in "abc":
            g.add_node(n)
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        g.add_edge("c", "a")
        wc = WeaklyConnected(g)
        assert wc.count() == wc.strong_count() == 1
        assert wc.one_way_barriers() == 0

    def test_direction_never_splits_a_weak_component(self):
        # edges pointing away from each other still join their endpoints
        g = Graph(directed=True)
        for n in "abc":
            g.add_node(n)
        g.add_edge("b", "a")
        g.add_edge("b", "c")
        assert WeaklyConnected(g).component_of("a") == {"a", "b", "c"}


class TestRefusals:
    def test_an_undirected_graph_is_refused(self):
        with pytest.raises(Invalid):
            WeaklyConnected(Graph())

    def test_a_missing_node_is_refused(self):
        with pytest.raises(Missing):
            WeaklyConnected(_one_way_chain()).component_of("ghost")


class TestReport:
    def test_the_note_states_both_counts_and_the_gap(self):
        note = WeaklyConnected(_one_way_chain()).note()
        assert "1 weak component(s) against 3 strong" in note
        assert "gap of 2" in note
