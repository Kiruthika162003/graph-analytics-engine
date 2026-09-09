from __future__ import annotations

import random
from itertools import combinations

import pytest

from mesh.errors import Invalid
from mesh.factories import complete, cycle, path, star, wheel
from mesh.graph import Graph
from mesh.tuttecount import TutteCount


class TestKnownCounts:
    def test_a_tree_has_one_spanning_tree_and_a_cycle_has_n(self):
        assert TutteCount(path(5)).trees == 1
        assert TutteCount(star(4)).trees == 1
        assert TutteCount(cycle(6)).trees == 6

    def test_a_complete_graph_follows_cayley(self):
        for n in (2, 3, 4, 5):
            assert TutteCount(complete(n)).trees == n ** (n - 2)

    def test_a_disconnected_graph_has_none(self):
        g = Graph()
        for n in "abcd":
            g.add_node(n)
        g.add_edge("a", "b")
        g.add_edge("c", "d")
        assert TutteCount(g).trees == 0

    def test_a_single_node_has_exactly_one(self):
        g = Graph()
        g.add_node("solo")
        assert TutteCount(g).trees == 1
        assert TutteCount(Graph()).trees == 1


class TestAgainstKirchhoff:
    def test_the_recursion_agrees_with_the_determinant_on_random_graphs(self):
        rng = random.Random(647)
        for _ in range(15):
            g = Graph()
            nodes = [str(i) for i in range(7)]
            for n in nodes:
                g.add_node(n)
            for a, b in combinations(nodes, 2):
                if rng.random() < 0.45:
                    g.add_edge(a, b)
            assert TutteCount(g).agrees_with_kirchhoff()

    def test_the_wheel_agrees_too(self):
        assert TutteCount(wheel(5)).agrees_with_kirchhoff()


class TestThroughAnEdge:
    def test_every_edge_of_a_cycle_lies_in_all_but_one_tree(self):
        tc = TutteCount(cycle(5))
        assert tc.trees_through("0", "1") == 4
        assert not tc.is_bridge("0", "1")

    def test_a_bridge_lies_in_every_tree(self):
        g = cycle(4)
        g.add_node("tail")
        g.add_edge("0", "tail")
        tc = TutteCount(g)
        assert tc.is_bridge("0", "tail")
        assert tc.trees_through("0", "tail") == tc.trees == 4

    def test_the_counts_through_each_edge_sum_to_trees_times_n_minus_one(self):
        # every spanning tree has n minus one edges, so summing over edges counts each
        # tree that many times
        g = complete(5)
        tc = TutteCount(g)
        total = sum(tc.trees_through(u, v) for u, v, _w in g.edges())
        assert total == tc.trees * (g.node_count() - 1)

    def test_an_absent_edge_is_refused(self):
        with pytest.raises(Invalid):
            TutteCount(path(3)).trees_through("0", "2")


class TestRefusal:
    def test_directed_and_oversized_graphs_are_refused(self):
        with pytest.raises(Invalid):
            TutteCount(Graph(directed=True))
        with pytest.raises(Invalid):
            TutteCount(path(13))


class TestReport:
    def test_the_note_states_the_count(self):
        note = TutteCount(cycle(4)).note()
        assert note.startswith("4 spanning tree(s) by deletion-contraction")
