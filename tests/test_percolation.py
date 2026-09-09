from __future__ import annotations

import random
from itertools import combinations

import pytest

from mesh.errors import Invalid
from mesh.factories import complete, cycle, grid, path
from mesh.graph import Graph
from mesh.percolation import Percolation


def _random_graph(seed: int, n: int, p: float) -> Graph:
    rng = random.Random(seed)
    g = Graph()
    nodes = [str(i) for i in range(n)]
    for node in nodes:
        g.add_node(node)
    for a, b in combinations(nodes, 2):
        if rng.random() < p:
            g.add_edge(a, b)
    return g


class TestEnds:
    def test_keeping_nothing_leaves_singletons_and_keeping_all_leaves_the_graph(self):
        pc = Percolation(cycle(8), seed=1)
        assert pc.giant_fraction(0.0, trials=3) == pytest.approx(1 / 8)
        assert pc.giant_fraction(1.0, trials=3) == pytest.approx(1.0)
        kept = pc.keep(1.0)
        assert kept.edge_count() == 8

    def test_the_largest_piece_of_a_kept_graph_is_counted_by_search(self):
        g = Graph()
        for n in "abcdef":
            g.add_node(n)
        for u, v in [("a", "b"), ("b", "c"), ("d", "e")]:
            g.add_edge(u, v)
        assert Percolation.largest_piece(g) == 3


class TestCurve:
    def test_the_averaged_curve_rises_with_retention(self):
        pc = Percolation(_random_graph(991, 40, 0.15), seed=2)
        curve = pc.sweep(steps=5, trials=30)
        fractions = [f for _p, f in curve]
        assert fractions == sorted(fractions)
        assert fractions[0] < 0.1
        assert fractions[-1] > 0.9

    def test_a_dense_graph_has_a_low_threshold_and_a_path_a_high_one(self):
        dense = Percolation(complete(20), seed=3).threshold_estimate(steps=10, trials=20)
        sparse = Percolation(path(20), seed=3).threshold_estimate(steps=10, trials=20)
        assert dense < sparse

    def test_molloy_reed_reads_one_over_b_on_a_regular_graph_with_degree_b_plus_one(self):
        # a 3-regular graph branches like a tree of branching 2, so the estimate is 1/2
        g = grid(4)
        pc = Percolation(g)
        assert 0 < pc.molloy_reed_threshold() < 1
        assert Percolation(cycle(10)).molloy_reed_threshold() == pytest.approx(1.0)
        assert Percolation(complete(5)).molloy_reed_threshold() == pytest.approx(1 / 3)


class TestSeeds:
    def test_the_same_seed_gives_the_same_curve(self):
        g = _random_graph(997, 30, 0.2)
        a = Percolation(g, seed=5).sweep(steps=4, trials=10)
        b = Percolation(g, seed=5).sweep(steps=4, trials=10)
        assert a == b


class TestRefusal:
    def test_bad_probabilities_counts_and_directed_graphs_are_refused(self):
        pc = Percolation(cycle(4))
        with pytest.raises(Invalid):
            pc.keep(1.5)
        with pytest.raises(Invalid):
            pc.giant_fraction(0.5, trials=0)
        with pytest.raises(Invalid):
            pc.sweep(steps=0)
        with pytest.raises(Invalid):
            Percolation(Graph(directed=True))

    def test_an_empty_graph_reads_zero(self):
        assert Percolation(Graph()).giant_fraction(0.5, trials=2) == 0.0


class TestReport:
    def test_the_note_prints_the_curve_and_both_thresholds(self):
        note = Percolation(cycle(6), seed=0).note(steps=2, trials=4)
        assert note.startswith("giant piece by retention: ")
        assert "threshold near" in note
        assert "Molloy-Reed 1.00" in note
