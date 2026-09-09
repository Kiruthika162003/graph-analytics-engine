from __future__ import annotations

import random
from itertools import combinations
from math import comb

import pytest

from mesh.errors import Invalid
from mesh.factories import complete, cycle, path, star
from mesh.graph import Graph
from mesh.quadcensus import SHAPES, QuadCensus


def _shape(edges: list[tuple[str, str]]) -> Graph:
    g = Graph()
    for n in "abcd":
        g.add_node(n)
    for u, v in edges:
        g.add_edge(u, v)
    return g


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


class TestSingleShapes:
    def test_each_of_the_six_shapes_is_named(self):
        cases = {
            "path": [("a", "b"), ("b", "c"), ("c", "d")],
            "claw": [("a", "b"), ("a", "c"), ("a", "d")],
            "cycle": [("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")],
            "paw": [("a", "b"), ("b", "c"), ("c", "a"), ("c", "d")],
            "diamond": [("a", "b"), ("b", "c"), ("c", "d"), ("d", "a"), ("a", "c")],
            "complete": [(u, v) for u, v in combinations("abcd", 2)],
        }
        for name, edges in cases.items():
            qc = QuadCensus(_shape(edges))
            assert qc.counts[name] == 1
            assert sum(qc.counts.values()) == 1

    def test_a_triangle_beside_an_isolated_node_is_not_connected(self):
        qc = QuadCensus(_shape([("a", "b"), ("b", "c"), ("c", "a")]))
        assert qc.disconnected == 1
        assert sum(qc.counts.values()) == 0
        assert qc.triangle_identity_holds()


class TestWholeGraphs:
    def test_a_complete_graph_is_all_complete_quads(self):
        qc = QuadCensus(complete(6))
        assert qc.counts["complete"] == comb(6, 4)
        assert qc.disconnected == 0

    def test_a_cycle_has_n_paths_and_a_star_has_choose_three_claws(self):
        assert QuadCensus(cycle(7)).counts["path"] == 7
        assert QuadCensus(star(5)).counts["claw"] == comb(5, 3)

    def test_counts_sum_to_n_choose_four_and_triangles_agree_on_random_graphs(self):
        for seed in range(1031, 1039):
            qc = QuadCensus(_random_graph(seed, 9, 0.45))
            assert qc.sums_to_choose_four()
            assert qc.triangle_identity_holds()
            assert set(qc.counts) == set(SHAPES)

    def test_a_small_graph_reads_its_triangles_directly(self):
        assert QuadCensus(cycle(3)).triangles_from_quads() == 1.0
        assert QuadCensus(path(3)).total() == 0


class TestRefusal:
    def test_directed_and_oversized_graphs_are_refused(self):
        with pytest.raises(Invalid):
            QuadCensus(Graph(directed=True))
        with pytest.raises(Invalid):
            QuadCensus(path(41))


class TestReport:
    def test_the_note_lists_the_shapes_present(self):
        note = QuadCensus(complete(4)).note()
        assert note == "1 quad(s): complete x1; 0 not connected"
        assert QuadCensus(path(4)).note() == "1 quad(s): path x1; 0 not connected"
