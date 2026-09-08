from __future__ import annotations

import random
from itertools import combinations

import pytest

from mesh.errors import Invalid
from mesh.havelhakimi import HavelHakimi


def _brute_graphical(degrees: list[int]) -> bool:
    # try every edge subset on n nodes and check whether one hits the degrees
    n = len(degrees)
    pairs = list(combinations(range(n), 2))
    for mask in range(1 << len(pairs)):
        deg = [0] * n
        for i, (a, b) in enumerate(pairs):
            if mask >> i & 1:
                deg[a] += 1
                deg[b] += 1
        if deg == degrees:
            return True
    return False


class TestGraphical:
    def test_a_triangle_sequence_is_realized(self):
        hh = HavelHakimi([2, 2, 2])
        assert hh.graphical
        assert hh.graph is not None
        assert hh.graph.edge_count() == 3

    def test_the_built_graph_has_exactly_the_requested_degrees(self):
        degrees = [3, 3, 2, 2, 2]
        hh = HavelHakimi(degrees)
        assert hh.graphical
        for i, d in enumerate(degrees):
            assert hh.graph.degree(str(i)) == d

    def test_all_zeros_is_the_empty_graph(self):
        hh = HavelHakimi([0, 0, 0])
        assert hh.graphical
        assert hh.graph.edge_count() == 0

    def test_a_star_sequence_is_realized(self):
        hh = HavelHakimi([3, 1, 1, 1])
        assert hh.graphical


class TestNotGraphical:
    def test_an_odd_sum_is_refused_by_parity(self):
        hh = HavelHakimi([1, 1, 1])
        assert not hh.graphical
        assert "odd" in hh.reason

    def test_a_degree_too_large_for_the_node_count_is_refused(self):
        hh = HavelHakimi([3, 1, 1])
        assert not hh.graphical

    def test_a_sequence_that_passes_parity_can_still_fail(self):
        # [3, 3, 1, 1] sums to 8, even, but the two 3s would each need three
        # of the other three nodes, leaving the 1s over-connected
        hh = HavelHakimi([3, 3, 1, 1])
        assert hh.parity_ok()
        assert not hh.graphical
        assert "only the full recursion caught it" in hh.note()


class TestRefusal:
    def test_a_negative_degree_is_refused(self):
        with pytest.raises(Invalid):
            HavelHakimi([2, -1])


class TestAgainstBruteForce:
    def test_the_verdict_matches_trying_every_edge_subset(self):
        rng = random.Random(131)
        for _ in range(40):
            n = rng.randint(1, 5)
            degrees = [rng.randint(0, n - 1) for _ in range(n)]
            hh = HavelHakimi(degrees)
            assert hh.graphical == _brute_graphical(degrees)
            if hh.graphical:
                for i, d in enumerate(degrees):
                    assert hh.graph.degree(str(i)) == d


class TestReport:
    def test_the_note_says_greedy_not_unique(self):
        assert "not unique" in HavelHakimi([2, 2, 2]).note()
