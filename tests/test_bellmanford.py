from __future__ import annotations

import random

import pytest

from mesh.bellmanford import BellmanFord
from mesh.dijkstra import Dijkstra
from mesh.errors import Missing, Negative, Unreachable
from mesh.graph import Graph


class TestNegativeEdges:
    def test_it_handles_a_negative_edge_dijkstra_could_not(self):
        # s->a=4, s->b=5, b->a=-3: best s->a is 5-3=2, not the direct 4
        g = Graph(directed=True)
        for n in "sab":
            g.add_node(n)
        g.add_edge("s", "a", 4)
        g.add_edge("s", "b", 5)
        g.add_edge("b", "a", -3)
        assert BellmanFord(g, "s").distance_to("a") == 2.0

    def test_a_negative_cycle_is_refused(self):
        g = Graph(directed=True)
        for n in "abc":
            g.add_node(n)
        g.add_edge("a", "b", 1)
        g.add_edge("b", "c", -3)
        g.add_edge("c", "a", 1)  # cycle sum 1-3+1 = -1
        with pytest.raises(Negative):
            BellmanFord(g, "a")


class TestRefusals:
    def test_a_missing_source_is_refused(self):
        g = Graph(directed=True)
        g.add_node("a")
        with pytest.raises(Missing):
            BellmanFord(g, "ghost")

    def test_an_unreachable_node_is_refused(self):
        g = Graph(directed=True)
        g.add_node("a")
        g.add_node("island")
        with pytest.raises(Unreachable):
            BellmanFord(g, "a").distance_to("island")


class TestAgainstDijkstra:
    def test_it_agrees_with_dijkstra_on_nonnegative_graphs(self):
        rng = random.Random(21)
        for _ in range(40):
            g = Graph(directed=True)
            nodes = [str(i) for i in range(7)]
            for n in nodes:
                g.add_node(n)
            for u in nodes:
                for v in nodes:
                    if u != v and rng.random() < 0.35:
                        g.add_edge(u, v, rng.randint(1, 9))
            bf = BellmanFord(g, "0")
            dij = Dijkstra(g, "0")
            for n in nodes:
                in_bf = n in bf.distance and bf.distance[n] != float("inf")
                in_dij = n in dij.distance
                assert in_bf == in_dij
                if in_bf:
                    assert bf.distance_to(n) == dij.distance_to(n)


class TestEarlyExit:
    def test_distances_settle_early_on_a_short_graph(self):
        # a two-edge chain settles in far fewer than node-1 passes
        g = Graph(directed=True)
        for n in "abc":
            g.add_node(n)
        g.add_edge("a", "b", 1)
        g.add_edge("b", "c", 1)
        bf = BellmanFord(g, "a")
        assert bf.settled_pass <= 2

    def test_the_note_reports_the_settling_pass(self):
        g = Graph(directed=True)
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b", 1)
        assert "settled after pass" in BellmanFord(g, "a").note()
