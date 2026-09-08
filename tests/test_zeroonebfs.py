from __future__ import annotations

import random
from itertools import pairwise

import pytest

from mesh.dijkstra import Dijkstra
from mesh.errors import Invalid, Missing, Unreachable
from mesh.graph import Graph
from mesh.zeroonebfs import ZeroOneBFS


def _free_and_paid() -> Graph:
    # s -1- a -0- b -1- t, and a direct s -1- t... no: make the free hop matter
    g = Graph(directed=True)
    for n in "sabt":
        g.add_node(n)
    g.add_edge("s", "a", 1)
    g.add_edge("a", "b", 0)
    g.add_edge("b", "t", 1)
    g.add_edge("s", "t", 1)
    g.add_edge("a", "t", 1)
    return g


class TestDistance:
    def test_zero_edges_cost_nothing(self):
        z = ZeroOneBFS(_free_and_paid(), "s")
        assert z.distance_to("b") == 1  # s-a costs 1, a-b is free

    def test_the_direct_paid_edge_ties_the_cheap_route(self):
        z = ZeroOneBFS(_free_and_paid(), "s")
        assert z.distance_to("t") == 1

    def test_a_later_zero_edge_improves_an_earlier_one_edge(self):
        # s -1- x, s -1- y -0- x: x should end at 1 either way, but y at 1
        # then x via the zero edge must not be recorded as 2
        g = Graph(directed=True)
        for n in "sxyz":
            g.add_node(n)
        g.add_edge("s", "x", 1)
        g.add_edge("s", "y", 1)
        g.add_edge("y", "z", 0)
        g.add_edge("x", "z", 1)
        assert ZeroOneBFS(g, "s").distance_to("z") == 1

    def test_the_path_steps_are_real_edges_summing_to_the_distance(self):
        g = _free_and_paid()
        z = ZeroOneBFS(g, "s")
        path = z.path_to("b")
        assert sum(g.weight(u, v) for u, v in pairwise(path)) == z.distance_to("b")


class TestRefusals:
    def test_a_weight_other_than_zero_or_one_is_refused(self):
        g = Graph(directed=True)
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b", 2)
        with pytest.raises(Invalid):
            ZeroOneBFS(g, "a")

    def test_a_missing_source_is_refused(self):
        with pytest.raises(Missing):
            ZeroOneBFS(_free_and_paid(), "ghost")

    def test_an_unreachable_node_is_refused(self):
        g = _free_and_paid()
        g.add_node("island")
        with pytest.raises(Unreachable):
            ZeroOneBFS(g, "s").distance_to("island")


class TestAgainstDijkstra:
    def test_distances_match_dijkstra_on_random_zero_one_graphs(self):
        rng = random.Random(197)
        for _ in range(40):
            directed = rng.random() < 0.5
            g = Graph(directed=directed)
            nodes = [str(i) for i in range(10)]
            for n in nodes:
                g.add_node(n)
            for u in nodes:
                for v in nodes:
                    if u < v and rng.random() < 0.3:
                        g.add_edge(u, v, rng.randint(0, 1))
            z = ZeroOneBFS(g, "0")
            d = Dijkstra(g, "0")
            for n in nodes:
                assert (n in z.distance) == (n in d.distance)
                if n in z.distance:
                    assert z.distance_to(n) == d.distance_to(n)


class TestReport:
    def test_the_note_counts_pushes(self):
        assert "push(es)" in ZeroOneBFS(_free_and_paid(), "s").note()
