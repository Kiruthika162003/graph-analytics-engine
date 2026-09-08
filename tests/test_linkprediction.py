from __future__ import annotations

import math

import pytest

from mesh.errors import Invalid, Missing
from mesh.graph import Graph
from mesh.linkprediction import LinkPrediction


def _triad_gap() -> Graph:
    # a and b both know c and d, but not each other; e is a loner linked to a
    g = Graph()
    for n in "abcde":
        g.add_node(n)
    for u, v in [("a", "c"), ("a", "d"), ("b", "c"), ("b", "d"), ("a", "e")]:
        g.add_edge(u, v)
    return g


class TestScores:
    def test_common_neighbors_counts_the_shared(self):
        assert LinkPrediction(_triad_gap()).common("a", "b") == 2

    def test_jaccard_divides_by_the_union(self):
        # a's neighbors {c,d,e}, b's {c,d}: shared 2, union 3
        assert LinkPrediction(_triad_gap()).jaccard("a", "b") == pytest.approx(2 / 3)

    def test_adamic_adar_weights_rare_neighbors_more(self):
        lp = LinkPrediction(_triad_gap())
        # c and d each have degree 2: two terms of 1 / log 2
        assert lp.adamic_adar("a", "b") == pytest.approx(2 / math.log(2))

    def test_preferential_attachment_multiplies_degrees(self):
        assert LinkPrediction(_triad_gap()).preferential("a", "b") == 3 * 2

    def test_a_pair_with_nothing_shared_scores_zero_on_structure(self):
        lp = LinkPrediction(_triad_gap())
        assert lp.common("e", "b") == 0
        assert lp.jaccard("e", "b") == 0
        assert lp.adamic_adar("e", "b") == 0


class TestRanking:
    def test_the_gap_in_the_triad_ranks_first(self):
        top = LinkPrediction(_triad_gap()).ranked("common")[0]
        assert (top[0], top[1]) == ("a", "b")

    def test_existing_edges_are_never_ranked(self):
        for u, v, _s in LinkPrediction(_triad_gap()).ranked("jaccard"):
            assert not _triad_gap().has_edge(u, v)

    def test_the_ranking_is_ordered_best_first(self):
        scores = [s for _u, _v, s in LinkPrediction(_triad_gap()).ranked("adamic_adar")]
        assert scores == sorted(scores, reverse=True)

    def test_hidden_edges_surface_near_the_top(self):
        # hide a-c from a dense triad and check it is the top prediction
        g = Graph()
        for n in "abcd":
            g.add_node(n)
        for u, v in [("a", "b"), ("a", "d"), ("b", "c"), ("b", "d"), ("c", "d")]:
            g.add_edge(u, v)
        top = LinkPrediction(g).ranked("adamic_adar")[0]
        assert (top[0], top[1]) == ("a", "c")


class TestRefusals:
    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            LinkPrediction(Graph(directed=True))

    def test_a_missing_node_is_refused(self):
        with pytest.raises(Missing):
            LinkPrediction(_triad_gap()).common("a", "ghost")

    def test_a_self_pair_is_refused(self):
        with pytest.raises(Invalid):
            LinkPrediction(_triad_gap()).jaccard("a", "a")

    def test_an_unknown_measure_is_refused(self):
        with pytest.raises(Invalid):
            LinkPrediction(_triad_gap()).score("a", "b", "magic")


class TestReport:
    def test_the_note_names_the_top_pair_and_its_basis(self):
        note = LinkPrediction(_triad_gap()).note("common")
        assert "top candidate a-b" in note
        assert "ranked by structure" in note
