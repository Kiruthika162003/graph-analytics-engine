from __future__ import annotations

import random
from itertools import combinations

import pytest

from mesh.errors import Invalid
from mesh.factories import cycle, path, star
from mesh.graph import Graph
from mesh.graphio import same_graph
from mesh.graphmerge import GraphMerge


def _random_graph(seed: int, n: int, p: float) -> Graph:
    rng = random.Random(seed)
    g = Graph()
    nodes = [str(i) for i in range(n)]
    for node in nodes:
        g.add_node(node)
    for a, b in combinations(nodes, 2):
        if rng.random() < p:
            g.add_edge(a, b, float(rng.randint(1, 5)))
    return g


class TestAlgebra:
    def test_union_with_the_empty_graph_and_intersection_with_itself(self):
        g = cycle(5)
        assert same_graph(GraphMerge(g, Graph()).union(), g)
        assert same_graph(GraphMerge(g, g, rule="first").intersection(), g)

    def test_the_difference_of_a_graph_from_itself_is_edgeless(self):
        g = star(4)
        d = GraphMerge(g, g).difference()
        assert d.node_count() == 5
        assert d.edge_count() == 0

    def test_the_edge_counts_obey_inclusion_exclusion_on_random_pairs(self):
        for seed in range(1013, 1023):
            a = _random_graph(seed, 7, 0.4)
            b = _random_graph(seed + 7, 7, 0.4)
            assert GraphMerge(a, b).counts_agree()

    def test_the_symmetric_difference_is_the_union_minus_the_intersection(self):
        a, b = path(4), cycle(4)
        gm = GraphMerge(a, b)
        assert gm.symmetric_difference().edge_count() == 1
        assert gm.symmetric_difference().has_edge("0", "3")


class TestWeightRules:
    def test_each_rule_resolves_a_shared_edge_its_own_way(self):
        a = Graph()
        b = Graph()
        for g in (a, b):
            g.add_node("x")
            g.add_node("y")
        a.add_edge("x", "y", 2.0)
        b.add_edge("y", "x", 5.0)
        assert GraphMerge(a, b, "sum").union().weight("x", "y") == 7.0
        assert GraphMerge(a, b, "max").union().weight("x", "y") == 5.0
        assert GraphMerge(a, b, "min").intersection().weight("x", "y") == 2.0
        assert GraphMerge(a, b, "first").intersection().weight("x", "y") == 2.0

    def test_directed_edges_keep_their_orientation_when_merging(self):
        a = Graph(directed=True)
        b = Graph(directed=True)
        for g in (a, b):
            g.add_node("p")
            g.add_node("q")
        a.add_edge("p", "q")
        b.add_edge("q", "p")
        u = GraphMerge(a, b).union()
        assert u.edge_count() == 2
        assert GraphMerge(a, b).intersection().edge_count() == 0


class TestRefusal:
    def test_mixed_direction_and_unknown_rules_are_refused(self):
        with pytest.raises(Invalid):
            GraphMerge(Graph(), Graph(directed=True))
        with pytest.raises(Invalid, match="no weight rule called 'mean'"):
            GraphMerge(Graph(), Graph(), rule="mean")


class TestReport:
    def test_the_note_lists_the_four_counts(self):
        note = GraphMerge(path(4), cycle(4)).note()
        assert note == "union 4, intersection 3, difference 0, symmetric 1 edge(s)"
