from __future__ import annotations

import random
from math import ceil, sqrt

import pytest

from mesh.burning import Burning
from mesh.errors import Invalid
from mesh.factories import complete, cycle, path, star
from mesh.graph import Graph
from mesh.prufer import Prufer


class TestKnownNumbers:
    def test_a_path_burns_in_ceiling_root_n_rounds(self):
        for n in (1, 2, 4, 5, 9, 10):
            b = Burning(path(n))
            assert len(b.exact()) == ceil(sqrt(n))

    def test_a_star_and_a_complete_graph_burn_in_two(self):
        assert len(Burning(star(6)).exact()) == 2
        assert len(Burning(complete(7)).exact()) == 2

    def test_a_single_node_burns_in_one(self):
        g = Graph()
        g.add_node("solo")
        assert Burning(g).exact() == ["solo"]

    def test_a_cycle_of_nine_burns_in_three(self):
        assert len(Burning(cycle(9)).exact()) == 3


class TestSequences:
    def test_a_sequence_burns_exactly_when_every_node_is_within_reach(self):
        b = Burning(path(4))
        assert b.burns(["1", "3"])
        assert not b.burns(["0", "3"])
        assert not b.burns(["1"])

    def test_greedy_sequences_burn_and_never_beat_the_exact_number(self):
        rng = random.Random(887)
        for _ in range(12):
            tree = Prufer.decode([rng.randrange(9) for _ in range(7)])
            b = Burning(tree)
            greedy = b.greedy()
            exact = b.exact()
            assert b.burns(greedy)
            assert b.burns(exact)
            assert len(greedy) >= len(exact)

    def test_the_root_n_conjecture_holds_on_random_trees(self):
        rng = random.Random(907)
        for _ in range(12):
            tree = Prufer.decode([rng.randrange(10) for _ in range(8)])
            b = Burning(tree)
            assert len(b.exact()) <= b.conjecture_bound()

    def test_a_fixed_round_count_returns_that_many_sources_even_when_short(self):
        b = Burning(path(9))
        two = b.greedy(rounds=2)
        assert len(two) == 2
        assert not b.burns(two)


class TestDisconnected:
    def test_pieces_burn_from_separate_sources(self):
        g = Graph()
        for n in "abcd":
            g.add_node(n)
        g.add_edge("a", "b")
        g.add_edge("c", "d")
        # the guess was two rounds; the second source has no reach left, so it burns
        # only itself and the other node of its pair needs a third round
        b = Burning(g)
        assert len(b.exact()) == 3
        assert b.burns(b.greedy())


class TestRefusal:
    def test_directed_and_oversized_graphs_are_refused(self):
        with pytest.raises(Invalid):
            Burning(Graph(directed=True))
        with pytest.raises(Invalid):
            Burning(path(13)).exact()


class TestReport:
    def test_the_note_states_rounds_and_the_bound(self):
        note = Burning(path(4)).note()
        assert note.startswith("burns in 2 round(s)")
        assert "root-n bound 2" in note
