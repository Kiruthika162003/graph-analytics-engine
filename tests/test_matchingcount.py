from __future__ import annotations

import random

import pytest

from mesh.errors import Invalid
from mesh.factories import complete, cycle, path
from mesh.graph import Graph
from mesh.graphproduct import GraphProduct
from mesh.matchingcount import PerfectMatchingCount


def _double_factorial(k: int) -> int:
    return 1 if k <= 0 else k * _double_factorial(k - 2)


def _random_bipartite(seed: int, n: int, p: float) -> Graph:
    rng = random.Random(seed)
    g = Graph()
    left = [f"l{i}" for i in range(n)]
    right = [f"r{i}" for i in range(n)]
    for node in left + right:
        g.add_node(node)
    for u in left:
        for v in right:
            if rng.random() < p:
                g.add_edge(u, v)
    return g


class TestKnownCounts:
    def test_a_complete_graph_has_the_double_factorial(self):
        for n in (2, 4, 6, 8):
            assert PerfectMatchingCount(complete(n)).count == _double_factorial(n - 1)

    def test_an_even_cycle_has_two_and_an_even_path_has_one(self):
        assert PerfectMatchingCount(cycle(8)).count == 2
        assert PerfectMatchingCount(path(6)).count == 1

    def test_an_odd_node_count_has_none(self):
        assert PerfectMatchingCount(cycle(5)).count == 0
        assert PerfectMatchingCount(path(3)).count == 0

    def test_a_ladder_counts_fibonacci(self):
        fib = [1, 1, 2, 3, 5, 8, 13, 21]
        for rungs in range(1, 8):
            ladder = GraphProduct(path(2), path(rungs)).cartesian()
            assert PerfectMatchingCount(ladder).count == fib[rungs]

    def test_the_empty_graph_has_one_empty_matching(self):
        assert PerfectMatchingCount(Graph()).count == 1


class TestPermanent:
    def test_the_recursion_agrees_with_ryser_on_random_bipartite_graphs(self):
        for seed in range(653, 668):
            g = _random_bipartite(seed, 5, 0.5)
            pm = PerfectMatchingCount(g)
            assert pm.count == pm.permanent()

    def test_the_permanent_of_complete_bipartite_is_a_factorial(self):
        g = _random_bipartite(0, 4, 1.1)
        assert PerfectMatchingCount(g).permanent() == 24

    def test_the_permanent_reading_refuses_an_odd_cycle(self):
        with pytest.raises(Invalid):
            PerfectMatchingCount(cycle(5)).permanent()

    def test_unequal_sides_give_zero(self):
        g = Graph()
        for n in ("a", "b", "x"):
            g.add_node(n)
        g.add_edge("a", "x")
        g.add_edge("b", "x")
        g.add_node("y")
        g.add_node("z")
        g.add_edge("a", "y")
        # left {a, b}, right {x, y, z}: z is isolated so nothing pairs it
        assert PerfectMatchingCount(g).permanent() == 0


class TestRefusal:
    def test_directed_and_oversized_graphs_are_refused(self):
        with pytest.raises(Invalid):
            PerfectMatchingCount(Graph(directed=True))
        with pytest.raises(Invalid):
            PerfectMatchingCount(path(18))


class TestReport:
    def test_the_note_states_the_count(self):
        assert PerfectMatchingCount(cycle(6)).note().startswith("2 perfect matching(s) over 6")
