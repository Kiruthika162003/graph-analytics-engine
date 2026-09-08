from __future__ import annotations

import random

import pytest

from mesh.dinic import Dinic
from mesh.edmondskarp import EdmondsKarp
from mesh.errors import Invalid, Missing
from mesh.graph import Graph
from mesh.pushrelabel import PushRelabel


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
        assert PushRelabel(_classic(), "s", "t").value == 23

    def test_a_bottleneck_chain_is_capped(self):
        g = Graph(directed=True)
        for n in "sat":
            g.add_node(n)
        g.add_edge("s", "a", 10)
        g.add_edge("a", "t", 3)
        assert PushRelabel(g, "s", "t").value == 3

    def test_excess_that_cannot_reach_the_sink_drains_home(self):
        # a dead end: s->a has capacity but a leads nowhere, so nothing flows
        g = Graph(directed=True)
        for n in "sat":
            g.add_node(n)
        g.add_edge("s", "a", 5)
        pr = PushRelabel(g, "s", "t")
        assert pr.value == 0
        assert pr.excess["a"] == 0

    def test_no_edges_means_zero_flow(self):
        g = Graph(directed=True)
        g.add_node("s")
        g.add_node("t")
        assert PushRelabel(g, "s", "t").value == 0


class TestFlowIsLegal:
    def test_capacity_and_conservation_hold_at_the_end(self):
        g = _classic()
        pr = PushRelabel(g, "s", "t")
        for u, v, cap in g.edges():
            # flow_on is the net flow across an antiparallel pair, so its floor
            # is minus the reverse capacity, not zero
            reverse_cap = g.weight(v, u) if g.has_edge(v, u) else 0
            assert -reverse_cap <= pr.flow_on(u, v) <= cap
        for node in "abcd":
            # net flow out of the node, counting an antiparallel pair once;
            # summing inflow and outflow separately double-counted such a
            # pair and only passed elsewhere when its net flow happened to be zero
            net_out = 0.0
            for other in g.nodes():
                if g.has_edge(node, other):
                    net_out += pr.flow_on(node, other)
                elif g.has_edge(other, node):
                    net_out -= pr.flow_on(other, node)
            assert net_out == 0
            assert pr.excess[node] == 0


class TestRefusals:
    def test_an_undirected_graph_is_refused(self):
        with pytest.raises(Invalid):
            PushRelabel(Graph(), "s", "t")

    def test_source_equal_to_sink_is_refused(self):
        g = Graph(directed=True)
        g.add_node("s")
        with pytest.raises(Invalid):
            PushRelabel(g, "s", "s")

    def test_a_missing_endpoint_is_refused(self):
        g = Graph(directed=True)
        g.add_node("s")
        with pytest.raises(Missing):
            PushRelabel(g, "s", "ghost")


class TestAgreesWithTheOtherTwo:
    def test_three_flow_algorithms_land_on_one_value(self):
        rng = random.Random(107)
        for _ in range(40):
            g = Graph(directed=True)
            nodes = ["s", "a", "b", "c", "d", "t"]
            for n in nodes:
                g.add_node(n)
            for u in nodes:
                for v in nodes:
                    if v not in (u, "s") and u != "t" and rng.random() < 0.45:
                        g.add_edge(u, v, rng.randint(1, 12))
            pr = PushRelabel(g, "s", "t").value
            assert pr == EdmondsKarp(g, "s", "t").value == Dinic(g, "s", "t").value


class TestReport:
    def test_the_note_states_pushes_and_relabels(self):
        note = PushRelabel(_classic(), "s", "t").note()
        assert "maximum flow 23" in note
        assert "relabel(s)" in note
