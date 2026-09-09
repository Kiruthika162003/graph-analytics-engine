from __future__ import annotations

import random
from itertools import combinations, permutations

import pytest

from mesh.errors import Invalid
from mesh.factories import complete, cycle, grid, path, star
from mesh.graph import Graph
from mesh.pathwidth import Pathwidth
from mesh.treewidth import TreewidthBound


def _binary_tree(depth: int) -> Graph:
    g = Graph()
    g.add_node("1")
    for i in range(2, 2**depth):
        g.add_node(str(i))
        g.add_edge(str(i // 2), str(i))
    return g


def _brute(g: Graph) -> int:
    pw = Pathwidth(g)
    return min(pw.separation(list(p)) for p in permutations(g.nodes()))


class TestKnownValues:
    def test_a_path_and_a_star_have_pathwidth_one(self):
        assert Pathwidth(path(7)).exact() == 1
        assert Pathwidth(star(5)).exact() == 1

    def test_a_cycle_has_two_and_a_complete_graph_n_minus_one(self):
        assert Pathwidth(cycle(6)).exact() == 2
        assert Pathwidth(complete(5)).exact() == 4

    def test_a_three_by_three_grid_has_pathwidth_three(self):
        assert Pathwidth(grid(3)).exact() == 3

    def test_a_binary_tree_of_depth_four_has_pathwidth_two_above_treewidth_one(self):
        tree = _binary_tree(4)
        assert Pathwidth(tree).exact() == 2
        assert TreewidthBound(tree).width == 1

    def test_an_edgeless_graph_has_pathwidth_zero(self):
        g = Graph()
        for n in "abc":
            g.add_node(n)
        assert Pathwidth(g).exact() == 0
        assert Pathwidth(Graph()).exact() == 0


class TestAgainstBruteForce:
    def test_the_subset_dynamic_matches_trying_every_ordering(self):
        rng = random.Random(877)
        for _ in range(10):
            g = Graph()
            nodes = [str(i) for i in range(6)]
            for n in nodes:
                g.add_node(n)
            for a, b in combinations(nodes, 2):
                if rng.random() < 0.45:
                    g.add_edge(a, b)
            assert Pathwidth(g).exact() == _brute(g)

    def test_the_level_order_bound_never_falls_below_the_exact_value(self):
        rng = random.Random(881)
        for _ in range(10):
            g = Graph()
            nodes = [str(i) for i in range(8)]
            for n in nodes:
                g.add_node(n)
            for a, b in combinations(nodes, 2):
                if rng.random() < 0.35:
                    g.add_edge(a, b)
            pw = Pathwidth(g)
            assert pw.level_order_bound() >= pw.exact()

    def test_pathwidth_is_at_least_the_treewidth_lower_bound(self):
        for g in (cycle(6), grid(3), star(4), path(5)):
            assert Pathwidth(g).exact() >= TreewidthBound(g).lower_bound()


class TestRefusal:
    def test_directed_and_oversized_graphs_are_refused(self):
        with pytest.raises(Invalid):
            Pathwidth(Graph(directed=True))
        with pytest.raises(Invalid):
            Pathwidth(path(17)).exact()


class TestReport:
    def test_the_note_compares_the_bound_with_the_exact_value(self):
        note = Pathwidth(path(5)).note()
        assert note == "pathwidth 1; the level-order bound of 1 meets the exact value"
