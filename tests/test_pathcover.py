from __future__ import annotations

import random
from itertools import pairwise, permutations

import pytest

from mesh.errors import Cyclic, Invalid
from mesh.graph import Graph
from mesh.pathcover import PathCover


def _brute_min_cover(g: Graph) -> int:
    # every ordering of the nodes, cut wherever consecutive nodes lack an edge
    best = len(g.nodes())
    for perm in permutations(g.nodes()):
        paths = 1
        for a, b in pairwise(perm):
            if not g.has_edge(a, b):
                paths += 1
        best = min(best, paths)
    return best


def _chain(k: int) -> Graph:
    g = Graph(directed=True)
    nodes = [str(i) for i in range(k)]
    for n in nodes:
        g.add_node(n)
    for i in range(k - 1):
        g.add_edge(nodes[i], nodes[i + 1])
    return g


class TestCover:
    def test_a_chain_is_one_path(self):
        pc = PathCover(_chain(5))
        assert pc.count() == 1
        assert pc.paths == [["0", "1", "2", "3", "4"]]

    def test_isolated_nodes_are_each_their_own_path(self):
        g = Graph(directed=True)
        for n in "abc":
            g.add_node(n)
        pc = PathCover(g)
        assert pc.count() == 3

    def test_a_fork_needs_two_paths(self):
        g = Graph(directed=True)
        for n in "abc":
            g.add_node(n)
        g.add_edge("a", "b")
        g.add_edge("a", "c")
        pc = PathCover(g)
        assert pc.count() == 2
        assert pc.covers_every_node_once()

    def test_every_path_follows_real_edges(self):
        g = _chain(4)
        g.add_node("x")
        g.add_edge("1", "x")
        pc = PathCover(g)
        for path in pc.paths:
            for a, b in pairwise(path):
                assert g.has_edge(a, b)

    def test_skipping_steps_can_need_fewer_chains(self):
        # a -> b -> c and a -> c: direct-step cover and closure cover agree;
        # but a -> b, a -> c with b, c incomparable: two chains either way;
        # the Dilworth count never exceeds the direct count
        g = Graph(directed=True)
        for n in "abcd":
            g.add_node(n)
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        g.add_edge("a", "d")
        pc = PathCover(g)
        assert pc.dilworth_count() <= pc.count()


class TestRefusals:
    def test_an_undirected_graph_is_refused(self):
        with pytest.raises(Invalid):
            PathCover(Graph())

    def test_a_cyclic_graph_is_refused(self):
        g = Graph(directed=True)
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b")
        g.add_edge("b", "a")
        with pytest.raises(Cyclic):
            PathCover(g)


class TestAgainstBruteForce:
    def test_the_count_matches_trying_every_ordering(self):
        rng = random.Random(449)
        for _ in range(30):
            g = Graph(directed=True)
            nodes = [str(i) for i in range(6)]
            for n in nodes:
                g.add_node(n)
            for i, u in enumerate(nodes):
                for v in nodes[i + 1 :]:
                    if rng.random() < 0.3:
                        g.add_edge(u, v)
            pc = PathCover(g)
            assert pc.covers_every_node_once()
            assert pc.count() == _brute_min_cover(g)


class TestReport:
    def test_the_note_states_both_counts(self):
        note = PathCover(_chain(4)).note()
        assert "1 path(s) cover 4 node(s)" in note
