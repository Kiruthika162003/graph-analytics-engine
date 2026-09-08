from __future__ import annotations

from itertools import combinations

import pytest

from mesh.errors import Invalid
from mesh.graph import Graph
from mesh.planarity import PlanarityCheck


def _complete(k: int) -> Graph:
    g = Graph()
    nodes = [str(i) for i in range(k)]
    for n in nodes:
        g.add_node(n)
    for a, b in combinations(nodes, 2):
        g.add_edge(a, b)
    return g


def _k33_named() -> Graph:
    g = Graph()
    for n in ["a", "b", "c", "x", "y", "z"]:
        g.add_node(n)
    for u in "abc":
        for v in "xyz":
            g.add_edge(u, v)
    return g


class TestObstructions:
    def test_k5_is_caught_by_the_euler_bound(self):
        pc = PlanarityCheck(_complete(5))
        assert pc.is_proven_non_planar()
        # 10 edges over a bound of 9: the bound decides before any search
        assert pc.obstruction == "Euler bound"

    def test_k33_is_caught_as_a_subgraph(self):
        pc = PlanarityCheck(_k33_named())
        assert pc.is_proven_non_planar()
        # 9 edges against a triangle-free bound of 8: also Euler, so make it
        # slip under by adding a node with no edges to raise the bound
        g = _k33_named()
        g.add_node("spare")
        pc2 = PlanarityCheck(g)
        assert pc2.is_proven_non_planar()
        assert pc2.obstruction == "K3,3 subgraph"

    def test_k5_hidden_inside_a_sparse_graph_is_found_by_search(self):
        g = _complete(5)
        for i in range(8):
            g.add_node(f"pad{i}")  # raises the Euler bound well above 10
        pc = PlanarityCheck(g)
        assert pc.obstruction == "K5 subgraph"

    def test_a_tree_shows_no_obstruction(self):
        g = Graph()
        for n in "abcd":
            g.add_node(n)
        for u, v in [("a", "b"), ("b", "c"), ("b", "d")]:
            g.add_edge(u, v)
        pc = PlanarityCheck(g)
        assert pc.verdict == "no obstruction found"
        assert not pc.is_proven_non_planar()

    def test_k4_is_planar_and_shows_nothing(self):
        assert PlanarityCheck(_complete(4)).verdict == "no obstruction found"


class TestEulerBound:
    def test_the_bound_is_three_n_minus_six_with_triangles(self):
        assert PlanarityCheck(_complete(4)).euler_bound() == 6

    def test_the_bound_tightens_for_triangle_free_graphs(self):
        # a 4-cycle has no triangle: 2n - 4 = 4
        g = Graph()
        for n in "abcd":
            g.add_node(n)
        for u, v in [("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")]:
            g.add_edge(u, v)
        assert PlanarityCheck(g).euler_bound() == 4


class TestRefusal:
    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            PlanarityCheck(Graph(directed=True))


class TestReport:
    def test_the_note_never_claims_planarity(self):
        note = PlanarityCheck(_complete(4)).note()
        assert "not a proof of planarity" in note
