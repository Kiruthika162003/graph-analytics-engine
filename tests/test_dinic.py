from __future__ import annotations

import random

import pytest

from mesh.dinic import Dinic
from mesh.edmondskarp import EdmondsKarp
from mesh.errors import Invalid, Missing
from mesh.graph import Graph


def _classic() -> Graph:
    g = Graph(directed=True)
    for n in ["s", "a", "b", "c", "d", "t"]:
        g.add_node(n)
    for u, v, cap in [
        ("s", "a", 16), ("s", "b", 13), ("a", "b", 10), ("b", "a", 4),
        ("a", "c", 12), ("c", "b", 9), ("b", "d", 14), ("d", "c", 7),
        ("c", "t", 20), ("d", "t", 4),
    ]:
        g.add_edge(u, v, cap)
    return g


class TestValue:
    def test_the_classic_network_flows_twenty_three(self):
        assert Dinic(_classic(), "s", "t").value == 23

    def test_a_bottleneck_chain_is_capped_by_its_narrowest_link(self):
        g = Graph(directed=True)
        for n in "sat":
            g.add_node(n)
        g.add_edge("s", "a", 10)
        g.add_edge("a", "t", 3)
        assert Dinic(g, "s", "t").value == 3

    def test_no_path_means_zero_flow_and_zero_phases(self):
        g = Graph(directed=True)
        g.add_node("s")
        g.add_node("t")
        d = Dinic(g, "s", "t")
        assert d.value == 0
        assert d.phases == 0


class TestConservation:
    def test_flow_respects_capacity_and_is_conserved(self):
        g = _classic()
        d = Dinic(g, "s", "t")
        for u, v, cap in g.edges():
            assert 0 <= d.flow_on(u, v) <= cap
        for node in "abcd":
            inflow = sum(d.flow_on(u, node) for u in g.nodes() if g.has_edge(u, node))
            outflow = sum(d.flow_on(node, v) for v in g.neighbors(node))
            assert inflow == outflow


class TestPhases:
    def test_phases_never_exceed_the_node_count(self):
        d = Dinic(_classic(), "s", "t")
        assert d.phases <= _classic().node_count()


class TestRefusals:
    def test_an_undirected_graph_is_refused(self):
        with pytest.raises(Invalid):
            Dinic(Graph(), "s", "t")

    def test_source_equal_to_sink_is_refused(self):
        g = Graph(directed=True)
        g.add_node("s")
        with pytest.raises(Invalid):
            Dinic(g, "s", "s")

    def test_a_missing_endpoint_is_refused(self):
        g = Graph(directed=True)
        g.add_node("s")
        with pytest.raises(Missing):
            Dinic(g, "s", "ghost")


class TestAgreesWithEdmondsKarp:
    def test_the_two_flow_algorithms_reach_the_same_value(self):
        rng = random.Random(101)
        for _ in range(40):
            g = Graph(directed=True)
            nodes = ["s", "a", "b", "c", "d", "t"]
            for n in nodes:
                g.add_node(n)
            for u in nodes:
                for v in nodes:
                    if v not in (u, "s") and u != "t" and rng.random() < 0.45:
                        g.add_edge(u, v, rng.randint(1, 12))
            assert Dinic(g, "s", "t").value == EdmondsKarp(g, "s", "t").value


class TestReport:
    def test_the_note_states_the_flow_and_phases(self):
        note = Dinic(_classic(), "s", "t").note()
        assert "maximum flow 23" in note
        assert "phase(s)" in note
