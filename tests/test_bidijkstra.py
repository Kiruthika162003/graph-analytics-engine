from __future__ import annotations

import random
from itertools import pairwise

import pytest

from mesh.bidijkstra import BidirectionalDijkstra
from mesh.dijkstra import Dijkstra
from mesh.errors import Invalid, Missing, Unreachable
from mesh.graph import Graph


def _cheap_crossing() -> Graph:
    # the classic trap: s-a and b-t are dear, a-b is cheap; the first node
    # settled by both sides is not on the shortest path
    g = Graph()
    for n in ["s", "a", "b", "t", "m"]:
        g.add_node(n)
    g.add_edge("s", "a", 4)
    g.add_edge("a", "b", 1)
    g.add_edge("b", "t", 4)
    g.add_edge("s", "m", 5)
    g.add_edge("m", "t", 5)
    return g


class TestDistance:
    def test_it_does_not_stop_at_the_first_common_node(self):
        bd = BidirectionalDijkstra(_cheap_crossing(), "s", "t")
        assert bd.distance() == 9  # s-a-b-t, not s-m-t at 10

    def test_source_equal_to_target_is_zero(self):
        bd = BidirectionalDijkstra(_cheap_crossing(), "s", "s")
        assert bd.distance() == 0
        assert bd.path() == ["s"]

    def test_the_path_is_a_real_walk_summing_to_the_distance(self):
        g = _cheap_crossing()
        bd = BidirectionalDijkstra(g, "s", "t")
        path = bd.path()
        assert path[0] == "s" and path[-1] == "t"
        assert sum(g.weight(u, v) for u, v in pairwise(path)) == bd.distance()

    def test_direction_is_respected_on_a_digraph(self):
        g = Graph(directed=True)
        for n in "abc":
            g.add_node(n)
        g.add_edge("a", "b", 1)
        g.add_edge("b", "c", 1)
        assert BidirectionalDijkstra(g, "a", "c").distance() == 2
        with pytest.raises(Unreachable):
            BidirectionalDijkstra(g, "c", "a").distance()


class TestRefusals:
    def test_a_negative_edge_is_refused(self):
        g = Graph(directed=True)
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b", -1)
        with pytest.raises(Invalid):
            BidirectionalDijkstra(g, "a", "b")

    def test_a_missing_endpoint_is_refused(self):
        with pytest.raises(Missing):
            BidirectionalDijkstra(_cheap_crossing(), "s", "ghost")

    def test_an_unreachable_target_is_refused(self):
        g = _cheap_crossing()
        g.add_node("island")
        with pytest.raises(Unreachable):
            BidirectionalDijkstra(g, "s", "island").distance()


class TestAgainstDijkstra:
    def test_distances_and_paths_match_on_random_weighted_graphs(self):
        rng = random.Random(211)
        for _ in range(40):
            directed = rng.random() < 0.5
            g = Graph(directed=directed)
            nodes = [str(i) for i in range(12)]
            for n in nodes:
                g.add_node(n)
            for u in nodes:
                for v in nodes:
                    if u != v and rng.random() < 0.2:
                        g.add_edge(u, v, rng.randint(1, 9))
            src, dst = rng.sample(nodes, 2)
            plain = Dijkstra(g, src)
            bd = BidirectionalDijkstra(g, src, dst)
            if dst in plain.distance:
                assert bd.distance() == plain.distance_to(dst)
                path = bd.path()
                assert sum(g.weight(u, v) for u, v in pairwise(path)) == bd.distance()
            else:
                with pytest.raises(Unreachable):
                    bd.distance()


class TestReport:
    def test_the_note_compares_settled_counts(self):
        assert "one-sided" in BidirectionalDijkstra(_cheap_crossing(), "s", "t").note()
