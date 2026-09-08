from __future__ import annotations

import random
from itertools import pairwise

import pytest

from mesh.dijkstra import Dijkstra
from mesh.errors import Invalid, Missing, Unreachable
from mesh.graph import Graph


def _weighted() -> Graph:
    g = Graph(directed=True)
    for n in "sabt":
        g.add_node(n)
    g.add_edge("s", "a", 1)
    g.add_edge("s", "b", 4)
    g.add_edge("a", "b", 1)
    g.add_edge("a", "t", 5)
    g.add_edge("b", "t", 1)
    return g


class TestDistance:
    def test_the_source_is_zero(self):
        assert Dijkstra(_weighted(), "s").distance_to("s") == 0.0

    def test_it_prefers_the_cheaper_multi_hop_path(self):
        # s->a->b->t costs 1+1+1=3, beats s->b->t (4+1=5) and s->a->t (1+5=6)
        assert Dijkstra(_weighted(), "s").distance_to("t") == 3.0

    def test_the_reconstructed_path_sums_to_the_distance(self):
        g = _weighted()
        d = Dijkstra(g, "s")
        path = d.path_to("t")
        total = sum(g.weight(u, v) for u, v in pairwise(path))
        assert total == d.distance_to("t")


class TestRefusals:
    def test_a_negative_edge_is_refused(self):
        g = Graph(directed=True)
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b", -2)
        with pytest.raises(Invalid) as caught:
            Dijkstra(g, "a")
        assert "Bellman-Ford" in str(caught.value)

    def test_a_missing_source_is_refused(self):
        with pytest.raises(Missing):
            Dijkstra(_weighted(), "ghost")

    def test_an_unreachable_target_is_refused(self):
        g = _weighted()
        g.add_node("island")
        with pytest.raises(Unreachable):
            Dijkstra(g, "s").distance_to("island")


class TestAgainstBruteForce:
    def test_it_matches_bellman_ford_where_weights_are_nonnegative(self):
        rng = random.Random(7)
        for _ in range(40):
            g = Graph(directed=True)
            nodes = [str(i) for i in range(8)]
            for n in nodes:
                g.add_node(n)
            for u in nodes:
                for v in nodes:
                    if u != v and rng.random() < 0.3:
                        g.add_edge(u, v, rng.randint(1, 9))
            src = "0"
            dij = Dijkstra(g, src)
            # brute force Bellman-Ford
            dist = {n: float("inf") for n in nodes}
            dist[src] = 0.0
            for _ in range(len(nodes) - 1):
                for u, v, w in g.edges():
                    dist[v] = min(dist[v], dist[u] + w)
            for n in nodes:
                if dist[n] == float("inf"):
                    assert n not in dij.distance
                else:
                    assert dij.distance_to(n) == dist[n]


class TestReport:
    def test_the_note_states_the_settled_count(self):
        assert "settled 4 node(s)" in Dijkstra(_weighted(), "s").note()
