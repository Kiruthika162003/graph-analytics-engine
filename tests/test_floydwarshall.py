from __future__ import annotations

import random
from itertools import pairwise

import pytest

from mesh.bellmanford import BellmanFord
from mesh.errors import Negative, Unreachable
from mesh.floydwarshall import FloydWarshall
from mesh.graph import Graph


def _small() -> Graph:
    g = Graph(directed=True)
    for n in "abcd":
        g.add_node(n)
    g.add_edge("a", "b", 3)
    g.add_edge("b", "c", 1)
    g.add_edge("a", "c", 5)
    g.add_edge("c", "d", 2)
    return g


class TestDistance:
    def test_a_multi_hop_beats_a_direct_edge(self):
        # a->b->c is 4, beats the direct a->c of 5
        assert FloydWarshall(_small()).distance("a", "c") == 4.0

    def test_a_node_reaches_itself_at_zero(self):
        assert FloydWarshall(_small()).distance("a", "a") == 0.0

    def test_the_path_sums_to_the_distance(self):
        g = _small()
        fw = FloydWarshall(g)
        path = fw.path("a", "d")
        total = sum(g.weight(u, v) for u, v in pairwise(path))
        assert total == fw.distance("a", "d")

    def test_an_unreachable_pair_is_refused(self):
        g = _small()
        g.add_node("island")
        with pytest.raises(Unreachable):
            FloydWarshall(g).distance("a", "island")


class TestUndirected:
    def test_an_undirected_edge_is_walkable_both_ways(self):
        # caught by the postman module: edges() lists an undirected edge once
        # and the reverse direction was left at infinity
        g = Graph()
        for n in "abc":
            g.add_node(n)
        g.add_edge("a", "b", 2)
        g.add_edge("b", "c", 3)
        fw = FloydWarshall(g)
        assert fw.distance("c", "a") == 5
        assert fw.distance("a", "c") == 5
        assert fw.path("c", "a") == ["c", "b", "a"]


class TestNegativeCycle:
    def test_a_negative_cycle_is_refused(self):
        g = Graph(directed=True)
        for n in "abc":
            g.add_node(n)
        g.add_edge("a", "b", 1)
        g.add_edge("b", "c", -4)
        g.add_edge("c", "a", 1)
        with pytest.raises(Negative):
            FloydWarshall(g)


class TestAgainstBellmanFord:
    def test_it_agrees_with_bellman_ford_from_every_source(self):
        rng = random.Random(33)
        for _ in range(30):
            g = Graph(directed=True)
            nodes = [str(i) for i in range(6)]
            for n in nodes:
                g.add_node(n)
            for u in nodes:
                for v in nodes:
                    if u != v and rng.random() < 0.35:
                        g.add_edge(u, v, rng.randint(-2, 9))
            try:
                fw = FloydWarshall(g)
            except Negative:
                continue  # skip graphs with a negative cycle
            for src in nodes:
                bf = BellmanFord(g, src)
                for dst in nodes:
                    reachable = bf.distance.get(dst, float("inf")) != float("inf")
                    if reachable:
                        assert fw.distance(src, dst) == bf.distance_to(dst)


class TestReport:
    def test_the_diameter_is_the_largest_finite_distance(self):
        # a->b->c->d = 3+1+2 = 6 is the longest shortest path
        assert FloydWarshall(_small()).diameter() == 6.0

    def test_the_note_states_the_diameter(self):
        assert "diameter 6" in FloydWarshall(_small()).note()
