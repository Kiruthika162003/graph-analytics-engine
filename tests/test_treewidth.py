from __future__ import annotations

from itertools import combinations

import pytest

from mesh.errors import Invalid
from mesh.graph import Graph
from mesh.treewidth import TreewidthBound


def _cycle(k: int) -> Graph:
    g = Graph()
    nodes = [str(i) for i in range(k)]
    for n in nodes:
        g.add_node(n)
    for i in range(k):
        g.add_edge(nodes[i], nodes[(i + 1) % k])
    return g


def _grid(n: int) -> Graph:
    g = Graph()
    for r in range(n):
        for c in range(n):
            g.add_node(f"{r},{c}")
    for r in range(n):
        for c in range(n):
            if c + 1 < n:
                g.add_edge(f"{r},{c}", f"{r},{c + 1}")
            if r + 1 < n:
                g.add_edge(f"{r},{c}", f"{r + 1},{c}")
    return g


class TestKnownWidths:
    def test_a_tree_has_width_one(self):
        g = Graph()
        for n in "abcde":
            g.add_node(n)
        for u, v in [("a", "b"), ("a", "c"), ("c", "d"), ("c", "e")]:
            g.add_edge(u, v)
        tw = TreewidthBound(g)
        assert tw.width == 1
        assert tw.is_exact()

    def test_a_cycle_has_width_two(self):
        tw = TreewidthBound(_cycle(7))
        assert tw.width == 2
        assert tw.is_exact()

    def test_a_complete_graph_has_width_n_minus_one(self):
        g = Graph()
        for n in "abcde":
            g.add_node(n)
        for a, b in combinations("abcde", 2):
            g.add_edge(a, b)
        tw = TreewidthBound(g)
        assert tw.width == 4
        assert tw.fill_edges == 0

    def test_a_three_by_three_grid_is_bounded_at_three(self):
        tw = TreewidthBound(_grid(3))
        assert tw.width == 3
        assert tw.lower_bound() <= 3

    def test_a_single_node_has_width_zero(self):
        g = Graph()
        g.add_node("solo")
        assert TreewidthBound(g).width == 0


class TestBoundDirection:
    def test_the_upper_bound_never_falls_below_the_degeneracy(self):
        for g in (_cycle(5), _grid(4), _cycle(3)):
            tw = TreewidthBound(g)
            assert tw.width >= tw.lower_bound()

    def test_eliminating_a_cycle_adds_fill(self):
        tw = TreewidthBound(_cycle(6))
        assert tw.fill_edges > 0

    def test_the_order_lists_every_node_once(self):
        g = _grid(3)
        assert sorted(TreewidthBound(g).order) == sorted(g.nodes())


class TestRefusals:
    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            TreewidthBound(Graph(directed=True))

    def test_an_empty_graph_is_refused(self):
        with pytest.raises(Invalid):
            TreewidthBound(Graph())


class TestReport:
    def test_the_note_states_both_bounds(self):
        note = TreewidthBound(_cycle(5)).note()
        assert "treewidth at most 2 and at least 2 (exact)" in note
