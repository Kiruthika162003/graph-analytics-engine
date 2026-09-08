from __future__ import annotations

import random

import pytest

from mesh.errors import Invalid
from mesh.permutationgraph import PermutationGraph


class TestBuild:
    def test_the_identity_has_no_edges_and_the_reverse_is_complete(self):
        assert PermutationGraph([0, 1, 2, 3]).graph.edge_count() == 0
        assert PermutationGraph([3, 2, 1, 0]).graph.edge_count() == 6

    def test_inversions_become_edges(self):
        pg = PermutationGraph([2, 0, 1])
        assert pg.graph.has_edge("0", "1")
        assert pg.graph.has_edge("0", "2")
        assert not pg.graph.has_edge("1", "2")
        assert pg.inversions() == 2


class TestNumbers:
    def test_the_runs_give_the_clique_and_independence_numbers(self):
        pg = PermutationGraph([3, 1, 4, 0, 2])
        assert pg.clique_number() == 3
        assert pg.independence_number() == 2
        assert pg.numbers_agree()

    def test_the_numbers_agree_with_the_graph_modules_on_random_permutations(self):
        rng = random.Random(587)
        for _ in range(30):
            perm = list(range(8))
            rng.shuffle(perm)
            assert PermutationGraph(perm).numbers_agree()

    def test_patience_sorting_finds_the_longest_increasing_run(self):
        assert PermutationGraph.longest_increasing([0, 8, 4, 12, 2, 10, 6, 14, 1, 9]) == 4
        assert PermutationGraph.longest_increasing([]) == 0


class TestComplement:
    def test_the_complement_is_the_reversed_permutation_on_random_inputs(self):
        rng = random.Random(593)
        for _ in range(20):
            perm = list(range(7))
            rng.shuffle(perm)
            assert PermutationGraph(perm).complement_is_the_reverse()


class TestRefusal:
    def test_a_repeated_or_missing_value_is_refused_by_name(self):
        with pytest.raises(Invalid, match="value 1 is repeated"):
            PermutationGraph([0, 1, 1])
        with pytest.raises(Invalid, match="value 1 is missing"):
            PermutationGraph([0, 2, 3])


class TestReport:
    def test_the_note_states_inversions_and_both_numbers(self):
        note = PermutationGraph([3, 1, 4, 0, 2]).note()
        assert "permutation of 5 with 6 inversion(s)" in note
        assert "clique number 3" in note
        assert "independence number 2" in note
