from __future__ import annotations

import random
from itertools import pairwise

import pytest

from mesh.bfs import BFS
from mesh.bidirectional import BidirectionalBFS
from mesh.errors import Missing, Unreachable
from mesh.graph import Graph


def _grid(n: int) -> Graph:
    g = Graph()
    for r in range(n):
        for c in range(n):
            g.add_node(f"{r},{c}")
    for r in range(n):
        for c in range(n):
            if c + 1 < n:
                g.add_edge(f"{r},{c}", f"{r},{c + 1}")
            if r + 1 < n:
                g.add_edge(f"{r},{c}", f"{r + 1},{c}")
    return g


class TestDistance:
    def test_it_finds_the_shortest_distance_on_a_grid(self):
        g = _grid(5)
        assert BidirectionalBFS(g, "0,0", "4,4").distance() == 8

    def test_source_equal_to_target_is_distance_zero(self):
        g = _grid(3)
        b = BidirectionalBFS(g, "1,1", "1,1")
        assert b.distance() == 0
        assert b.path() == ["1,1"]

    def test_the_path_is_a_real_walk_of_the_right_length(self):
        g = _grid(4)
        b = BidirectionalBFS(g, "0,0", "3,3")
        path = b.path()
        assert path[0] == "0,0" and path[-1] == "3,3"
        assert len(path) == b.distance() + 1
        for u, v in pairwise(path):
            assert g.has_edge(u, v)

    def test_it_respects_direction_on_a_digraph(self):
        g = Graph(directed=True)
        for n in "abc":
            g.add_node(n)
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        assert BidirectionalBFS(g, "a", "c").distance() == 2
        with pytest.raises(Unreachable):
            BidirectionalBFS(g, "c", "a").distance()


class TestSaving:
    def test_it_expands_fewer_nodes_than_one_sided_bfs_on_a_grid(self):
        g = _grid(9)
        b = BidirectionalBFS(g, "0,0", "8,8")
        assert b.expanded < b.plain_expanded()


class TestRefusals:
    def test_a_missing_endpoint_is_refused(self):
        with pytest.raises(Missing):
            BidirectionalBFS(_grid(2), "0,0", "ghost")

    def test_an_unreachable_target_is_refused(self):
        g = _grid(2)
        g.add_node("island")
        with pytest.raises(Unreachable):
            BidirectionalBFS(g, "0,0", "island").distance()


class TestAgainstBFS:
    def test_distances_match_plain_bfs_on_random_graphs(self):
        rng = random.Random(37)
        for _ in range(30):
            directed = rng.random() < 0.5
            g = Graph(directed=directed)
            nodes = [str(i) for i in range(10)]
            for n in nodes:
                g.add_node(n)
            for u in nodes:
                for v in nodes:
                    if u != v and rng.random() < 0.2:
                        g.add_edge(u, v)
            src, dst = rng.sample(nodes, 2)
            plain = BFS(g, src)
            bi = BidirectionalBFS(g, src, dst)
            if dst in plain.distance:
                assert bi.distance() == plain.distance_to(dst)
            else:
                with pytest.raises(Unreachable):
                    bi.distance()


class TestReport:
    def test_the_note_compares_expansion_counts(self):
        assert "against" in BidirectionalBFS(_grid(4), "0,0", "3,3").note()
