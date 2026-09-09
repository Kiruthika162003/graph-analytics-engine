from __future__ import annotations

import random
from itertools import combinations

import pytest

from mesh.errors import Invalid
from mesh.factories import complete, cycle, grid, path, star, wheel
from mesh.feedbackvertexset import FeedbackVertexSet
from mesh.graph import Graph


def _two_cycles_sharing(node: str) -> Graph:
    g = Graph()
    for n in ("a", "b", node, "x", "y"):
        g.add_node(n)
    for u, v in [("a", "b"), ("b", node), (node, "a"), ("x", "y"), ("y", node), (node, "x")]:
        g.add_edge(u, v)
    return g


class TestKnownSizes:
    def test_trees_need_nothing(self):
        assert FeedbackVertexSet(path(6)).exact() == set()
        assert FeedbackVertexSet(star(4)).greedy() == set()

    def test_a_cycle_needs_one_and_two_cycles_sharing_a_node_need_that_node(self):
        assert len(FeedbackVertexSet(cycle(7)).exact()) == 1
        fvs = FeedbackVertexSet(_two_cycles_sharing("hub"))
        assert fvs.exact() == {"hub"}
        assert fvs.greedy() == {"hub"}

    def test_a_wheel_needs_the_hub_and_one_rim_node(self):
        fvs = FeedbackVertexSet(wheel(6))
        exact = fvs.exact()
        assert len(exact) == 2
        assert "hub" in exact
        assert fvs.breaks_every_cycle(exact)

    def test_a_complete_graph_needs_n_minus_two(self):
        assert len(FeedbackVertexSet(complete(6)).exact()) == 4

    def test_a_three_by_three_grid_needs_two(self):
        assert len(FeedbackVertexSet(grid(3)).exact()) == 2


class TestGreedyAgainstExact:
    def test_greedy_is_a_feedback_set_never_smaller_than_exact_on_random_graphs(self):
        rng = random.Random(883)
        for _ in range(12):
            g = Graph()
            nodes = [str(i) for i in range(8)]
            for n in nodes:
                g.add_node(n)
            for a, b in combinations(nodes, 2):
                if rng.random() < 0.3:
                    g.add_edge(a, b)
            fvs = FeedbackVertexSet(g)
            exact, greedy = fvs.exact(), fvs.greedy()
            assert fvs.breaks_every_cycle(exact)
            assert fvs.breaks_every_cycle(greedy)
            assert len(greedy) >= len(exact)

    def test_removing_one_fewer_node_than_the_exact_set_leaves_a_cycle(self):
        fvs = FeedbackVertexSet(wheel(5))
        exact = fvs.exact()
        for node in exact:
            assert not fvs.breaks_every_cycle(exact - {node})


class TestRefusal:
    def test_directed_and_oversized_graphs_are_refused(self):
        with pytest.raises(Invalid):
            FeedbackVertexSet(Graph(directed=True))
        with pytest.raises(Invalid):
            FeedbackVertexSet(path(15)).exact()

    def test_an_empty_graph_needs_nothing(self):
        assert FeedbackVertexSet(Graph()).exact() == set()


class TestReport:
    def test_the_note_gives_both_sizes_and_the_gap(self):
        note = FeedbackVertexSet(cycle(5)).note()
        assert "of size 1; greedy took 1, 0 above the exact size" in note
