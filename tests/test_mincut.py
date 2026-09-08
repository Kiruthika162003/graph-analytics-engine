from __future__ import annotations

import random

import pytest

from mesh.errors import Invalid
from mesh.graph import Graph
from mesh.mincut import MinCut


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


class TestCut:
    def test_the_cut_capacity_equals_the_flow(self):
        mc = MinCut(_classic(), "s", "t")
        assert mc.value == 23
        assert mc.matches_flow()

    def test_the_source_is_on_the_source_side_and_the_sink_is_not(self):
        mc = MinCut(_classic(), "s", "t")
        assert "s" in mc.source_side
        assert "t" not in mc.source_side

    def test_every_cut_edge_crosses_from_source_side_to_sink_side(self):
        mc = MinCut(_classic(), "s", "t")
        for u, v, _w in mc.edges:
            assert u in mc.source_side and v not in mc.source_side

    def test_every_cut_edge_is_saturated(self):
        assert MinCut(_classic(), "s", "t").every_cut_edge_is_saturated()

    def test_the_classic_network_cuts_at_the_known_edges(self):
        mc = MinCut(_classic(), "s", "t")
        assert {(u, v) for u, v, _w in mc.edges} == {("a", "c"), ("d", "c"), ("d", "t")}

    def test_a_single_bottleneck_edge_is_the_whole_cut(self):
        g = Graph(directed=True)
        for n in "sat":
            g.add_node(n)
        g.add_edge("s", "a", 10)
        g.add_edge("a", "t", 3)
        mc = MinCut(g, "s", "t")
        assert mc.edges == [("a", "t", 3)]


class TestRefusal:
    def test_an_undirected_graph_is_refused_through_the_flow(self):
        with pytest.raises(Invalid):
            MinCut(Graph(), "s", "t")


class TestAgainstBruteForce:
    def test_no_cut_is_cheaper_than_the_one_returned(self):
        rng = random.Random(103)
        for _ in range(25):
            g = Graph(directed=True)
            nodes = ["s", "a", "b", "c", "t"]
            for n in nodes:
                g.add_node(n)
            for u in nodes:
                for v in nodes:
                    if v not in (u, "s") and u != "t" and rng.random() < 0.5:
                        g.add_edge(u, v, rng.randint(1, 10))
            mc = MinCut(g, "s", "t")
            assert mc.matches_flow()
            interior = ["a", "b", "c"]
            for mask in range(1 << 3):
                side = {"s"} | {interior[i] for i in range(3) if mask >> i & 1}
                cut = sum(w for u, v, w in g.edges() if u in side and v not in side)
                assert cut >= mc.capacity()


class TestReport:
    def test_the_note_states_edges_and_capacity(self):
        note = MinCut(_classic(), "s", "t").note()
        assert "cut of 3 edge(s)" in note
        assert "capacity 23" in note
