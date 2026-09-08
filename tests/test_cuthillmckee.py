from __future__ import annotations

import random
from itertools import combinations

import pytest

from mesh.cuthillmckee import ReverseCuthillMcKee
from mesh.errors import Invalid
from mesh.graph import Graph


def _scrambled_path(n: int, seed: int) -> Graph:
    # a path whose nodes are inserted in shuffled order, so the original
    # numbering has a large bandwidth
    rng = random.Random(seed)
    labels = [str(i) for i in range(n)]
    order = labels[:]
    rng.shuffle(order)
    g = Graph()
    for node in order:
        g.add_node(node)
    for i in range(n - 1):
        g.add_edge(labels[i], labels[i + 1])
    return g


class TestOrdering:
    def test_a_scrambled_path_drops_to_bandwidth_one(self):
        rcm = ReverseCuthillMcKee(_scrambled_path(20, seed=3))
        assert rcm.after == 1
        assert rcm.before > 1

    def test_the_order_is_a_permutation_of_every_node(self):
        g = _scrambled_path(10, seed=4)
        rcm = ReverseCuthillMcKee(g)
        assert sorted(rcm.order) == sorted(g.nodes())

    def test_a_complete_graph_cannot_improve(self):
        g = Graph()
        for n in "abcde":
            g.add_node(n)
        for a, b in combinations("abcde", 2):
            g.add_edge(a, b)
        rcm = ReverseCuthillMcKee(g)
        assert rcm.before == rcm.after == 4

    def test_every_component_is_placed(self):
        g = _scrambled_path(6, seed=5)
        g.add_node("x")
        g.add_node("y")
        g.add_edge("x", "y")
        rcm = ReverseCuthillMcKee(g)
        assert len(rcm.order) == 8

    def test_bandwidth_is_the_largest_positional_gap(self):
        g = Graph()
        for n in "abc":
            g.add_node(n)
        g.add_edge("a", "c")
        assert ReverseCuthillMcKee(g).bandwidth(["a", "b", "c"]) == 2


class TestNeverWorse:
    def test_the_ordering_never_exceeds_the_original_bandwidth_on_grids(self):
        for n in (3, 4, 5):
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
            rcm = ReverseCuthillMcKee(g)
            assert rcm.after <= rcm.before
            assert rcm.after <= 2 * n


class TestRefusals:
    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            ReverseCuthillMcKee(Graph(directed=True))

    def test_an_empty_graph_is_refused(self):
        with pytest.raises(Invalid):
            ReverseCuthillMcKee(Graph())


class TestReport:
    def test_the_note_states_before_and_after(self):
        note = ReverseCuthillMcKee(_scrambled_path(12, seed=6)).note()
        assert "to 1 (ratio" in note
