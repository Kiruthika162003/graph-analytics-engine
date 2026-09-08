from __future__ import annotations

from math import log2

import pytest

from mesh.entropy import GraphEntropy
from mesh.errors import Invalid
from mesh.factories import complete, cycle, path, star
from mesh.graph import Graph


def _two_cliques() -> Graph:
    g = Graph()
    for group in ("abcd", "wxyz"):
        for n in group:
            g.add_node(n)
        for i, a in enumerate(group):
            for b in group[i + 1 :]:
                g.add_edge(a, b)
    g.add_edge("d", "w")
    return g


class TestDegreeEntropy:
    def test_a_regular_graph_has_zero_degree_entropy(self):
        assert GraphEntropy(cycle(6)).degree_entropy() == 0.0
        assert GraphEntropy(complete(5)).degree_entropy() == 0.0

    def test_a_star_splits_between_hub_and_leaves(self):
        h = GraphEntropy(star(3)).degree_entropy()
        expected = -(0.25 * log2(0.25) + 0.75 * log2(0.75))
        assert h == pytest.approx(expected)

    def test_an_empty_graph_scores_zero(self):
        assert GraphEntropy(Graph()).degree_entropy() == 0.0


class TestWalkEntropyRate:
    def test_a_regular_graph_yields_log_of_the_degree(self):
        assert GraphEntropy(cycle(7)).walk_entropy_rate() == pytest.approx(1.0)
        assert GraphEntropy(complete(5)).walk_entropy_rate() == pytest.approx(2.0)

    def test_a_path_charges_only_at_the_inner_nodes(self):
        # ends have one choice and contribute nothing; n-2 inner nodes each log 2
        n = 6
        rate = GraphEntropy(path(n)).walk_entropy_rate()
        assert rate == pytest.approx((n - 2) / (n - 1))

    def test_weights_skew_the_choice_and_lower_the_rate(self):
        even = Graph()
        skewed = Graph()
        for g, w in ((even, 1.0), (skewed, 9.0)):
            for n in "abc":
                g.add_node(n)
            g.add_edge("a", "b", w)
            g.add_edge("a", "c", 1.0)
        assert GraphEntropy(skewed).walk_entropy_rate() < GraphEntropy(even).walk_entropy_rate()

    def test_no_edges_means_no_steps_and_no_bits(self):
        g = Graph()
        g.add_node("a")
        assert GraphEntropy(g).walk_entropy_rate() == 0.0


class TestStructuralEntropy:
    def test_the_one_dimensional_reading_is_log_n_on_a_regular_graph(self):
        assert GraphEntropy(cycle(8)).structural_entropy() == pytest.approx(3.0)

    def test_trivial_and_singleton_partitions_both_give_the_one_dimensional_value(self):
        g = _two_cliques()
        e = GraphEntropy(g)
        one = e.structural_entropy()
        assert e.partition_entropy([g.nodes()]) == pytest.approx(one)
        assert e.partition_entropy([[n] for n in g.nodes()]) == pytest.approx(one)

    def test_the_real_communities_score_below_the_trivial_partition(self):
        g = _two_cliques()
        e = GraphEntropy(g)
        good = e.partition_entropy([list("abcd"), list("wxyz")])
        assert good < e.structural_entropy()
        # a partition that cuts across the cliques does worse than the real one
        bad = e.partition_entropy([list("abwx"), list("cdyz")])
        assert bad > good


class TestRefusal:
    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            GraphEntropy(Graph(directed=True))

    def test_a_partition_that_repeats_or_misses_a_node_is_refused_by_name(self):
        e = GraphEntropy(path(3))
        with pytest.raises(Invalid, match="'1' appears in two"):
            e.partition_entropy([["0", "1"], ["1", "2"]])
        with pytest.raises(Invalid, match="'2' is in no"):
            e.partition_entropy([["0", "1"]])
        with pytest.raises(Invalid, match="'q' is not a node"):
            e.partition_entropy([["0", "1", "2", "q"]])


class TestReport:
    def test_the_note_carries_all_three_readings_in_bits(self):
        note = GraphEntropy(cycle(4)).note()
        assert "degree entropy 0.000 bits" in note
        assert "walk entropy rate 1.000 bits per step" in note
        assert "structural entropy 2.000 bits" in note
