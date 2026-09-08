from __future__ import annotations

import random
from itertools import combinations

import pytest

from mesh.errors import Invalid
from mesh.graph import Graph
from mesh.tournament import Tournament


def _random_tournament(rng: random.Random, n: int) -> Graph:
    g = Graph(directed=True)
    names = [f"p{i}" for i in range(n)]
    for name in names:
        g.add_node(name)
    for a, b in combinations(names, 2):
        if rng.random() < 0.5:
            g.add_edge(a, b)
        else:
            g.add_edge(b, a)
    return g


def _transitive(n: int) -> Graph:
    g = Graph(directed=True)
    names = [f"t{i}" for i in range(n)]
    for name in names:
        g.add_node(name)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            g.add_edge(a, b)
    return g


def _rock_paper_scissors() -> Graph:
    g = Graph(directed=True)
    for n in ("rock", "paper", "scissors"):
        g.add_node(n)
    g.add_edge("rock", "scissors")
    g.add_edge("scissors", "paper")
    g.add_edge("paper", "rock")
    return g


class TestShape:
    def test_an_undirected_graph_is_refused(self):
        with pytest.raises(Invalid):
            Tournament(Graph())

    def test_a_missing_pair_and_a_double_pair_are_refused_by_name(self):
        g = Graph(directed=True)
        for n in "abc":
            g.add_node(n)
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        with pytest.raises(Invalid, match="'a' and 'c' never play"):
            Tournament(g)
        g.add_edge("a", "c")
        g.add_edge("c", "a")
        with pytest.raises(Invalid, match="'a' and 'c' play each other twice"):
            Tournament(g)


class TestScores:
    def test_a_transitive_tournament_scores_zero_through_n_minus_one(self):
        t = Tournament(_transitive(5))
        assert t.score_sequence() == [0, 1, 2, 3, 4]
        assert t.is_transitive()

    def test_rock_paper_scissors_is_a_cycle_with_equal_scores(self):
        t = Tournament(_rock_paper_scissors())
        assert t.score_sequence() == [1, 1, 1]
        assert not t.is_transitive()

    def test_landau_holds_for_every_real_score_sequence(self):
        rng = random.Random(479)
        for _ in range(30):
            t = Tournament(_random_tournament(rng, 7))
            assert Tournament.landau_holds(t.score_sequence())

    def test_landau_rejects_sequences_no_tournament_produces(self):
        assert not Tournament.landau_holds([0, 0, 3])
        assert not Tournament.landau_holds([2, 2, 2, 2])
        assert Tournament.landau_holds([1, 1, 1])
        assert Tournament.landau_holds([])


class TestHamiltonianPath:
    def test_insertion_builds_a_path_along_the_arcs_in_every_tournament(self):
        rng = random.Random(487)
        for _ in range(40):
            t = Tournament(_random_tournament(rng, 8))
            assert t.path_walks_arcs()

    def test_the_transitive_path_is_the_ranking(self):
        t = Tournament(_transitive(4))
        assert t.hamiltonian_path() == ["t0", "t1", "t2", "t3"]


class TestKings:
    def test_the_top_scorer_is_always_a_king(self):
        rng = random.Random(491)
        for _ in range(40):
            t = Tournament(_random_tournament(rng, 7))
            assert t.top_scorer() in t.kings()

    def test_every_player_is_a_king_in_rock_paper_scissors(self):
        kings = sorted(Tournament(_rock_paper_scissors()).kings())
        assert kings == ["paper", "rock", "scissors"]

    def test_a_transitive_tournament_has_exactly_one_king(self):
        assert Tournament(_transitive(6)).kings() == ["t0"]


class TestReport:
    def test_the_note_states_scores_kings_and_transitivity(self):
        note = Tournament(_transitive(3)).note()
        assert "scores [0, 1, 2], 1 king(s), top scorer t0, transitive" in note
        assert "upset cycle" in Tournament(_rock_paper_scissors()).note()
