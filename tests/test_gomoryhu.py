from __future__ import annotations

import random
from itertools import combinations

import pytest

from mesh.errors import Invalid, Missing
from mesh.factories import complete, cycle, path
from mesh.gomoryhu import GomoryHu
from mesh.graph import Graph
from mesh.mincut import MinCut


def _random_weighted(seed: int, n: int, p: float) -> Graph:
    rng = random.Random(seed)
    g = Graph()
    nodes = [str(i) for i in range(n)]
    for node in nodes:
        g.add_node(node)
    for a, b in combinations(nodes, 2):
        if rng.random() < p:
            g.add_edge(a, b, float(rng.randint(1, 9)))
    return g


class TestTree:
    def test_the_tree_has_one_edge_per_node_beyond_the_first(self):
        gh = GomoryHu(cycle(6))
        assert gh.tree().edge_count() == 5
        assert gh.flows == 5

    def test_a_cycle_has_every_min_cut_equal_to_two(self):
        gh = GomoryHu(cycle(6))
        for a, b in combinations(cycle(6).nodes(), 2):
            assert gh.min_cut(a, b) == 2.0

    def test_a_path_cuts_at_the_lightest_edge_between_the_two(self):
        g = path(4)
        for u, v, w in [("0", "1", 5.0), ("1", "2", 2.0), ("2", "3", 7.0)]:
            g.add_edge(u, v, w)
        gh = GomoryHu(g)
        assert gh.min_cut("0", "3") == 2.0
        assert gh.min_cut("0", "1") == 5.0
        assert gh.min_cut("2", "3") == 7.0

    def test_a_complete_graph_cuts_at_the_degree(self):
        gh = GomoryHu(complete(5))
        assert gh.min_cut("0", "4") == 4.0


class TestAgainstFlow:
    def test_every_pair_agrees_with_a_direct_flow_on_random_graphs(self):
        for seed in (503, 509, 521):
            g = _random_weighted(seed, 8, 0.5)
            gh = GomoryHu(g)
            for a, b in combinations(g.nodes(), 2):
                assert gh.min_cut(a, b) == MinCut(gh.flow_graph, a, b).value
                assert gh.agrees_with_flow(a, b)

    def test_every_tree_edge_is_a_real_cut_of_its_own_weight(self):
        # without Gusfield's swap the tree is only an equivalent flow tree, whose
        # values are right but whose edges need not be cuts; the swap makes them cuts
        for seed in (541, 547):
            g = _random_weighted(seed, 8, 0.5)
            gh = GomoryHu(g)
            for node in gh.parent:
                assert gh.edge_is_a_real_cut(node)

    def test_the_tree_holds_at_most_n_minus_one_distinct_values(self):
        g = _random_weighted(523, 9, 0.6)
        gh = GomoryHu(g)
        values = {gh.min_cut(a, b) for a, b in combinations(g.nodes(), 2)}
        assert len(values) <= g.node_count() - 1

    def test_a_disconnected_pair_has_a_zero_cut(self):
        g = Graph()
        for n in "abcd":
            g.add_node(n)
        g.add_edge("a", "b", 3.0)
        g.add_edge("c", "d", 4.0)
        gh = GomoryHu(g)
        assert gh.min_cut("a", "c") == 0.0
        assert gh.min_cut("a", "b") == 3.0


class TestQueries:
    def test_a_node_against_itself_is_zero(self):
        assert GomoryHu(cycle(4)).min_cut("1", "1") == 0.0

    def test_an_absent_node_is_refused_by_name(self):
        with pytest.raises(Missing, match="'zz'"):
            GomoryHu(cycle(4)).min_cut("0", "zz")

    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            GomoryHu(Graph(directed=True))

    def test_an_empty_graph_builds_an_empty_tree(self):
        gh = GomoryHu(Graph())
        assert gh.tree().node_count() == 0
        assert gh.flows == 0


class TestReport:
    def test_the_note_counts_flows_and_lists_the_distinct_values(self):
        note = GomoryHu(cycle(5)).note()
        assert "over 5 node(s) from 4 flow(s)" in note
        assert "distinct cut values [2.0]" in note
