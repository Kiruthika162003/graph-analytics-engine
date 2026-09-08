from __future__ import annotations

import random
from itertools import permutations

import pytest

from mesh.errors import Invalid
from mesh.hungarian import Hungarian


def _brute_min(cost: list[list[float]]) -> float:
    n = len(cost)
    return min(sum(cost[w][perm[w]] for w in range(n)) for perm in permutations(range(n)))


class TestAssignment:
    def test_the_obvious_diagonal_is_chosen(self):
        cost = [[1, 9, 9], [9, 1, 9], [9, 9, 1]]
        h = Hungarian(cost)
        assert h.assignment == [0, 1, 2]
        assert h.total == 3

    def test_the_one_to_one_constraint_forces_a_tradeoff(self):
        # both workers prefer job 0; one must take job 1
        cost = [[1, 5], [2, 9]]
        h = Hungarian(cost)
        # w0->j1 (5) + w1->j0 (2) = 7 beats w0->j0 (1) + w1->j1 (9) = 10
        assert h.total == 7
        assert h.assignment == [1, 0]

    def test_every_job_is_used_exactly_once(self):
        cost = [[4, 2, 8], [4, 3, 7], [3, 1, 6]]
        h = Hungarian(cost)
        assert sorted(h.assignment) == [0, 1, 2]

    def test_a_single_cell_matrix(self):
        assert Hungarian([[7]]).total == 7


class TestBounds:
    def test_the_total_never_falls_below_the_row_minima(self):
        rng = random.Random(3)
        for _ in range(30):
            n = rng.randint(2, 5)
            cost = [[rng.randint(0, 20) for _ in range(n)] for _ in range(n)]
            h = Hungarian(cost)
            assert h.total >= h.row_minima_bound()

    def test_the_bound_is_met_when_cheapest_jobs_are_distinct(self):
        cost = [[1, 9, 9], [9, 2, 9], [9, 9, 3]]
        h = Hungarian(cost)
        assert h.total == h.row_minima_bound()


class TestRefusals:
    def test_a_non_square_matrix_is_refused(self):
        with pytest.raises(Invalid):
            Hungarian([[1, 2], [3, 4], [5, 6]])

    def test_an_empty_matrix_is_refused(self):
        with pytest.raises(Invalid):
            Hungarian([])


class TestAgainstBruteForce:
    def test_the_total_matches_trying_every_permutation(self):
        rng = random.Random(29)
        for _ in range(40):
            n = rng.randint(1, 6)
            cost = [[rng.randint(0, 30) for _ in range(n)] for _ in range(n)]
            h = Hungarian(cost)
            assert h.total == _brute_min(cost)
            # and the reported assignment really sums to that total
            assert sum(cost[w][h.assignment[w]] for w in range(n)) == h.total


class TestReport:
    def test_the_note_states_the_total_and_bound(self):
        note = Hungarian([[1, 5], [2, 9]]).note()
        assert "optimal total 7" in note
        assert "row-minima bound of 3" in note
