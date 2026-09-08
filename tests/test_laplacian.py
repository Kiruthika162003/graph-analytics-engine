from __future__ import annotations

import random
from itertools import combinations

import pytest

from mesh.connectedcomponents import ConnectedComponents
from mesh.errors import Invalid
from mesh.graph import Graph
from mesh.laplacian import Laplacian


def _path(k: int) -> Graph:
    g = Graph()
    nodes = [str(i) for i in range(k)]
    for n in nodes:
        g.add_node(n)
    for i in range(k - 1):
        g.add_edge(nodes[i], nodes[i + 1])
    return g


class TestMatrix:
    def test_the_diagonal_holds_degrees_and_off_diagonal_minus_edges(self):
        lap = Laplacian(_path(3)).matrix
        assert lap == [[1.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 1.0]]

    def test_every_row_sums_to_zero(self):
        assert Laplacian(_path(5)).row_sums_vanish()

    def test_the_quadratic_form_equals_the_edge_difference_sum(self):
        g = _path(4)
        g.add_edge("0", "3", 2.5)
        lap = Laplacian(g)
        x = {"0": 1.0, "1": -2.0, "2": 0.5, "3": 3.0}
        assert lap.quadratic_form(x) == pytest.approx(lap.edge_difference_sum(x))

    def test_the_normalized_form_has_ones_on_the_diagonal(self):
        norm = Laplacian(_path(3)).normalized()
        assert [norm[i][i] for i in range(3)] == [1.0, 1.0, 1.0]
        assert norm[0][1] == pytest.approx(-1 / (1 * 2) ** 0.5)


class TestNullity:
    def test_a_connected_graph_has_nullity_one(self):
        assert Laplacian(_path(6)).nullity() == 1

    def test_nullity_counts_the_components(self):
        g = _path(3)
        g.add_node("x")
        g.add_node("y")
        g.add_edge("x", "y")
        g.add_node("lonely")
        assert Laplacian(g).nullity() == 3

    def test_nullity_matches_union_find_on_random_graphs(self):
        rng = random.Random(283)
        for _ in range(30):
            g = Graph()
            nodes = [str(i) for i in range(9)]
            for n in nodes:
                g.add_node(n)
            for a, b in combinations(nodes, 2):
                if rng.random() < 0.15:
                    g.add_edge(a, b, rng.randint(1, 5))
            assert Laplacian(g).nullity() == ConnectedComponents(g).count()


class TestRefusals:
    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            Laplacian(Graph(directed=True))

    def test_an_empty_graph_is_refused(self):
        with pytest.raises(Invalid):
            Laplacian(Graph())


class TestReport:
    def test_the_note_says_the_two_methods_agree(self):
        assert "agree" in Laplacian(_path(4)).note()
