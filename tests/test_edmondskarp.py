from __future__ import annotations

import random

import pytest

from mesh.edmondskarp import EdmondsKarp
from mesh.errors import Invalid, Missing
from mesh.graph import Graph


def _classic() -> Graph:
    # the textbook network: max flow from s to t is 23
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
        assert EdmondsKarp(_classic(), "s", "t").value == 23

    def test_a_single_edge_carries_its_capacity(self):
        g = Graph(directed=True)
        g.add_node("s")
        g.add_node("t")
        g.add_edge("s", "t", 7)
        assert EdmondsKarp(g, "s", "t").value == 7

    def test_a_chain_is_limited_by_its_bottleneck(self):
        g = Graph(directed=True)
        for n in "sat":
            g.add_node(n)
        g.add_edge("s", "a", 10)
        g.add_edge("a", "t", 3)  # the narrow link
        assert EdmondsKarp(g, "s", "t").value == 3

    def test_no_path_means_zero_flow(self):
        g = Graph(directed=True)
        g.add_node("s")
        g.add_node("t")
        assert EdmondsKarp(g, "s", "t").value == 0


class TestConservation:
    def test_flow_never_exceeds_capacity_and_is_conserved(self):
        g = _classic()
        ek = EdmondsKarp(g, "s", "t")
        for u, v, cap in g.edges():
            # flow_on is the net flow across an antiparallel pair, so its floor
            # is minus the reverse capacity; a zero floor passed here only by
            # augmentation-order luck until push-relabel broke it
            reverse_cap = g.weight(v, u) if g.has_edge(v, u) else 0
            assert -reverse_cap <= ek.flow_on(u, v) <= cap
        # every interior node passes on exactly what it receives: net flow out,
        # counting an antiparallel pair once rather than in both sums
        for node in "abcd":
            net_out = 0.0
            for other in g.nodes():
                if g.has_edge(node, other):
                    net_out += ek.flow_on(node, other)
                elif g.has_edge(other, node):
                    net_out -= ek.flow_on(other, node)
            assert net_out == 0


class TestRefusals:
    def test_an_undirected_graph_is_refused(self):
        with pytest.raises(Invalid):
            EdmondsKarp(Graph(), "s", "t")

    def test_source_equal_to_sink_is_refused(self):
        g = Graph(directed=True)
        g.add_node("s")
        with pytest.raises(Invalid):
            EdmondsKarp(g, "s", "s")

    def test_a_missing_endpoint_is_refused(self):
        g = Graph(directed=True)
        g.add_node("s")
        with pytest.raises(Missing):
            EdmondsKarp(g, "s", "ghost")

    def test_a_negative_capacity_is_refused(self):
        g = Graph(directed=True)
        g.add_node("s")
        g.add_node("t")
        g.add_edge("s", "t", -1)
        with pytest.raises(Invalid):
            EdmondsKarp(g, "s", "t")


class TestAgainstBruteForce:
    def test_flow_equals_the_minimum_cut_on_random_networks(self):
        # max-flow equals min-cut: enumerate every source-side subset
        rng = random.Random(9)
        for _ in range(25):
            g = Graph(directed=True)
            nodes = ["s", "a", "b", "c", "t"]
            for n in nodes:
                g.add_node(n)
            for u in nodes:
                for v in nodes:
                    # no edges into the source or out of the sink
                    if v not in (u, "s") and u != "t" and rng.random() < 0.5:
                        g.add_edge(u, v, rng.randint(1, 10))
            flow = EdmondsKarp(g, "s", "t").value
            interior = ["a", "b", "c"]
            best_cut = float("inf")
            for mask in range(1 << len(interior)):
                side = {"s"} | {interior[i] for i in range(3) if mask >> i & 1}
                cut = sum(w for u, v, w in g.edges() if u in side and v not in side)
                best_cut = min(best_cut, cut)
            assert flow == best_cut


class TestReport:
    def test_the_note_states_the_flow_value(self):
        assert "maximum flow 23" in EdmondsKarp(_classic(), "s", "t").note()
