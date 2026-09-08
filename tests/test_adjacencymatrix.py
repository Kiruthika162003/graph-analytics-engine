from __future__ import annotations

import random
from itertools import combinations

import pytest

from mesh.adjacencymatrix import AdjacencyMatrix
from mesh.errors import Invalid, Missing
from mesh.graph import Graph
from mesh.triangles import Triangles


def _triangle() -> Graph:
    g = Graph()
    for n in "abc":
        g.add_node(n)
    for a, b in combinations("abc", 2):
        g.add_edge(a, b)
    return g


def _brute_walks(g: Graph, u: str, v: str, length: int) -> int:
    if length == 0:
        return int(u == v)
    return sum(_brute_walks(g, w, v, length - 1) for w in g.neighbors(u))


class TestMatrix:
    def test_an_undirected_matrix_is_symmetric(self):
        assert AdjacencyMatrix(_triangle()).is_symmetric()

    def test_a_directed_matrix_is_one_sided(self):
        g = Graph(directed=True)
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b")
        m = AdjacencyMatrix(g)
        assert m.matrix == [[0, 1], [0, 0]]
        assert not m.is_symmetric()

    def test_the_zeroth_power_is_the_identity(self):
        m = AdjacencyMatrix(_triangle())
        assert m.power(0) == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    def test_a_negative_power_is_refused(self):
        with pytest.raises(Invalid):
            AdjacencyMatrix(_triangle()).power(-1)


class TestWalks:
    def test_two_step_walks_around_a_triangle(self):
        # from a: a-b-a and a-c-a return, a-b-c and a-c-b reach... count them
        m = AdjacencyMatrix(_triangle())
        assert m.walks("a", "a", 2) == 2
        assert m.walks("a", "b", 2) == 1

    def test_a_missing_node_is_refused(self):
        with pytest.raises(Missing):
            AdjacencyMatrix(_triangle()).walks("a", "ghost", 1)

    def test_walk_counts_match_recursive_enumeration(self):
        rng = random.Random(43)
        for _ in range(20):
            directed = rng.random() < 0.5
            g = Graph(directed=directed)
            nodes = [str(i) for i in range(6)]
            for n in nodes:
                g.add_node(n)
            for u in nodes:
                for v in nodes:
                    if u != v and rng.random() < 0.35:
                        g.add_edge(u, v)
            m = AdjacencyMatrix(g)
            for length in range(0, 5):
                u, v = rng.choice(nodes), rng.choice(nodes)
                assert m.walks(u, v, length) == _brute_walks(g, u, v, length)


class TestTriangles:
    def test_the_cube_trace_counts_triangles(self):
        assert AdjacencyMatrix(_triangle()).triangles() == 1

    def test_it_agrees_with_neighborhood_intersection_counting(self):
        rng = random.Random(47)
        for _ in range(20):
            g = Graph()
            nodes = [str(i) for i in range(8)]
            for n in nodes:
                g.add_node(n)
            for a, b in combinations(nodes, 2):
                if rng.random() < 0.4:
                    g.add_edge(a, b)
            assert AdjacencyMatrix(g).triangles() == Triangles(g).total

    def test_the_triangle_count_is_refused_for_a_digraph(self):
        with pytest.raises(Invalid):
            AdjacencyMatrix(Graph(directed=True)).triangles()


class TestReport:
    def test_density_is_ones_over_cells(self):
        # a triangle has 6 ones in 9 cells
        assert AdjacencyMatrix(_triangle()).density() == pytest.approx(6 / 9)

    def test_the_note_states_the_size(self):
        assert "3x3 matrix" in AdjacencyMatrix(_triangle()).note()
