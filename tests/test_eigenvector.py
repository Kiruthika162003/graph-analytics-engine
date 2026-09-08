from __future__ import annotations

import math
from itertools import combinations

import pytest

from mesh.eigenvector import EigenvectorCentrality
from mesh.errors import Invalid
from mesh.graph import Graph


def _complete(k: int) -> Graph:
    g = Graph()
    nodes = [str(i) for i in range(k)]
    for n in nodes:
        g.add_node(n)
    for a, b in combinations(nodes, 2):
        g.add_edge(a, b)
    return g


def _hub_and_wheel() -> Graph:
    # a triangle core a,b,c fully connected, plus d attached only to a
    g = Graph()
    for n in "abcd":
        g.add_node(n)
    for x, y in combinations("abc", 2):
        g.add_edge(x, y)
    g.add_edge("a", "d")
    return g


class TestScores:
    def test_a_complete_graph_scores_everyone_equally(self):
        ec = EigenvectorCentrality(_complete(4))
        values = list(ec.score.values())
        assert max(values) - min(values) < 1e-8

    def test_the_vector_has_unit_length(self):
        ec = EigenvectorCentrality(_hub_and_wheel())
        assert math.sqrt(sum(x * x for x in ec.score.values())) == pytest.approx(1.0)

    def test_the_node_with_important_neighbors_ranks_highest(self):
        ec = EigenvectorCentrality(_hub_and_wheel())
        assert ec.top(1)[0][0] == "a"
        # d has one neighbor, the best one, yet ranks below the core
        assert ec.score["d"] < ec.score["b"]

    def test_the_result_is_an_eigenvector(self):
        g = _hub_and_wheel()
        ec = EigenvectorCentrality(g, tolerance=1e-13)
        for n in g.nodes():
            image = sum(ec.score[m] for m in g.neighbors(n))
            assert image == pytest.approx(ec.eigenvalue * ec.score[n], abs=1e-8)


class TestEigenvalue:
    def test_a_complete_graph_has_eigenvalue_k_minus_one(self):
        ec = EigenvectorCentrality(_complete(5))
        assert ec.eigenvalue == pytest.approx(4.0)

    def test_the_eigenvalue_sits_between_average_and_max_degree(self):
        ec = EigenvectorCentrality(_hub_and_wheel())
        low, high = ec.degree_bounds()
        assert low - 1e-9 <= ec.eigenvalue <= high + 1e-9


class TestRefusals:
    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            EigenvectorCentrality(Graph(directed=True))

    def test_a_disconnected_graph_is_refused(self):
        g = _complete(3)
        g.add_node("island")
        with pytest.raises(Invalid):
            EigenvectorCentrality(g)

    def test_a_bipartite_graph_is_refused(self):
        g = Graph()
        for n in "abcd":
            g.add_node(n)
        for u, v in [("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")]:
            g.add_edge(u, v)
        with pytest.raises(Invalid) as caught:
            EigenvectorCentrality(g)
        assert "alternate" in str(caught.value)


class TestReport:
    def test_the_note_names_the_top_node_and_eigenvalue(self):
        note = EigenvectorCentrality(_hub_and_wheel()).note()
        assert "top 'a'" in note
        assert "eigenvalue" in note
