from __future__ import annotations

import pytest

from mesh.factories import complete, cycle, path, star
from mesh.graph import Graph
from mesh.harmonic import Harmonic


class TestClosedForms:
    def test_a_star_hub_scores_one_and_leaves_score_the_form(self):
        n = 6
        h = Harmonic(star(n - 1))
        assert h.scores["0"] == pytest.approx(1.0)
        assert h.scores["1"] == pytest.approx((1 + (n - 2) / 2) / (n - 1))
        assert h.top() == "0"

    def test_every_node_of_a_complete_graph_scores_one(self):
        assert all(s == pytest.approx(1.0) for s in Harmonic(complete(5)).scores.values())

    def test_a_path_end_scores_the_harmonic_sum(self):
        h = Harmonic(path(4))
        assert h.scores["0"] == pytest.approx((1 + 1 / 2 + 1 / 3) / 3)
        assert h.top() in ("1", "2")


class TestDisconnected:
    def test_unreachable_nodes_count_zero_while_closeness_collapses(self):
        g = Graph()
        for n in "abcd":
            g.add_node(n)
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        h = Harmonic(g)
        assert h.scores["b"] == pytest.approx(2 / 3)
        assert h.scores["d"] == 0.0
        assert h.closeness("b") == 0.0

    def test_a_single_node_scores_zero_without_dividing(self):
        g = Graph()
        g.add_node("solo")
        h = Harmonic(g)
        assert h.scores["solo"] == 0.0
        assert h.closeness("solo") == 0.0


class TestDirectedAndWeighted:
    def test_arcs_are_followed_outward(self):
        g = Graph(directed=True)
        for n in "abc":
            g.add_node(n)
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        h = Harmonic(g)
        assert h.scores["a"] == pytest.approx((1 + 1 / 2) / 2)
        assert h.scores["c"] == 0.0

    def test_weights_change_the_reciprocals(self):
        g = path(3)
        g.add_edge("0", "1", 4.0)
        plain = Harmonic(g).scores["0"]
        weighted = Harmonic(g, weighted=True).scores["0"]
        assert weighted < plain


class TestRankings:
    def test_rankings_agree_on_a_cycle_and_the_note_says_so(self):
        h = Harmonic(cycle(6))
        assert h.rankings_agree()
        assert "agrees with closeness" in h.note()

    def test_the_empty_graph_has_nothing_to_rank(self):
        h = Harmonic(Graph())
        assert h.top() is None
        assert h.note() == "no nodes to score"
