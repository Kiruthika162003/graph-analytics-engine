from __future__ import annotations

import random

import pytest

from mesh.condensation import Condensation
from mesh.errors import Invalid, Missing
from mesh.graph import Graph
from mesh.toposort import TopologicalSort
from mesh.transitiveclosure import TransitiveClosure


def _two_cycles_linked() -> Graph:
    # {a,b} cycle -> {c,d} cycle, plus a lone sink e fed by d
    g = Graph(directed=True)
    for n in "abcde":
        g.add_node(n)
    for u, v in [("a", "b"), ("b", "a"), ("c", "d"), ("d", "c"), ("b", "c"), ("d", "e")]:
        g.add_edge(u, v)
    return g


class TestStructure:
    def test_components_become_the_dag_nodes(self):
        c = Condensation(_two_cycles_linked())
        assert len(c.components) == 3
        assert c.dag.node_count() == 3

    def test_members_of_a_cycle_share_a_component(self):
        c = Condensation(_two_cycles_linked())
        assert c.component_of("a") == c.component_of("b")
        assert c.component_of("a") != c.component_of("c")

    def test_the_condensation_is_acyclic(self):
        c = Condensation(_two_cycles_linked())
        assert TopologicalSort(c.dag).is_acyclic()

    def test_sources_and_sinks_are_found(self):
        c = Condensation(_two_cycles_linked())
        assert c.sources() == [c.component_of("a")]
        assert c.sinks() == [c.component_of("e")]

    def test_the_order_puts_upstream_components_first(self):
        c = Condensation(_two_cycles_linked())
        order = c.order()
        assert order.index(c.component_of("a")) < order.index(c.component_of("c"))
        assert order.index(c.component_of("c")) < order.index(c.component_of("e"))


class TestReachability:
    def test_reaches_within_a_component(self):
        assert Condensation(_two_cycles_linked()).reaches("a", "b")

    def test_reaches_downstream_but_not_upstream(self):
        c = Condensation(_two_cycles_linked())
        assert c.reaches("a", "e")
        assert not c.reaches("e", "a")


class TestRefusals:
    def test_an_undirected_graph_is_refused(self):
        with pytest.raises(Invalid):
            Condensation(Graph())

    def test_a_missing_node_is_refused(self):
        with pytest.raises(Missing):
            Condensation(_two_cycles_linked()).component_of("ghost")


class TestAgainstClosure:
    def test_component_reachability_matches_node_reachability(self):
        rng = random.Random(59)
        for _ in range(30):
            g = Graph(directed=True)
            nodes = [str(i) for i in range(9)]
            for n in nodes:
                g.add_node(n)
            for u in nodes:
                for v in nodes:
                    if u != v and rng.random() < 0.2:
                        g.add_edge(u, v)
            c = Condensation(g)
            tc = TransitiveClosure(g)
            assert TopologicalSort(c.dag).is_acyclic()
            for u in nodes:
                for v in nodes:
                    expected = u == v or tc.reaches(u, v)
                    assert c.reaches(u, v) == expected


class TestReport:
    def test_the_note_states_components_and_sources(self):
        note = Condensation(_two_cycles_linked()).note()
        assert "3 component(s) from 5 node(s)" in note
        assert "1 source(s)" in note
