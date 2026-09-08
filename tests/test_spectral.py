from __future__ import annotations

import math
from itertools import combinations

import pytest

from mesh.errors import Invalid
from mesh.graph import Graph
from mesh.spectral import SpectralBisection


def _two_cliques_bridged() -> Graph:
    g = Graph()
    left = [f"l{i}" for i in range(4)]
    right = [f"r{i}" for i in range(4)]
    for n in left + right:
        g.add_node(n)
    for a, b in combinations(left, 2):
        g.add_edge(a, b)
    for a, b in combinations(right, 2):
        g.add_edge(a, b)
    g.add_edge("l0", "r0")
    return g


def _path(k: int) -> Graph:
    g = Graph()
    nodes = [str(i) for i in range(k)]
    for n in nodes:
        g.add_node(n)
    for i in range(k - 1):
        g.add_edge(nodes[i], nodes[i + 1])
    return g


class TestBisection:
    def test_the_bridge_is_cut_and_the_cliques_are_the_sides(self):
        sb = SpectralBisection(_two_cliques_bridged())
        pos, neg = sb.sides()
        left = {f"l{i}" for i in range(4)}
        assert left in (pos, neg)
        assert sb.edges_cut() == 1

    def test_a_path_splits_at_its_middle(self):
        sb = SpectralBisection(_path(6))
        pos, neg = sb.sides()
        assert {len(pos), len(neg)} == {3}
        assert sb.edges_cut() == 1

    def test_the_fiedler_vector_is_orthogonal_to_the_constant(self):
        sb = SpectralBisection(_two_cliques_bridged())
        assert abs(sum(sb.fiedler.values())) < 1e-6

    def test_the_fiedler_vector_is_an_eigenvector(self):
        g = _path(5)
        sb = SpectralBisection(g, tolerance=1e-13)
        lam = sb.algebraic_connectivity
        for n in g.nodes():
            image = g.degree(n) * sb.fiedler[n] - sum(sb.fiedler[m] for m in g.neighbors(n))
            assert image == pytest.approx(lam * sb.fiedler[n], abs=1e-6)


class TestAlgebraicConnectivity:
    def test_a_complete_graph_has_connectivity_equal_to_its_size(self):
        g = Graph()
        for n in "abcd":
            g.add_node(n)
        for a, b in combinations("abcd", 2):
            g.add_edge(a, b)
        assert SpectralBisection(g).algebraic_connectivity == pytest.approx(4.0, abs=1e-6)

    def test_a_bridged_graph_is_far_less_connected_than_a_clique(self):
        weak = SpectralBisection(_two_cliques_bridged()).algebraic_connectivity
        assert weak < 1.0

    def test_a_path_has_the_known_closed_form(self):
        # for a path of n nodes the value is 2 - 2 cos(pi / n)
        sb = SpectralBisection(_path(6), tolerance=1e-13)
        expected = 2 - 2 * math.cos(math.pi / 6)
        assert sb.algebraic_connectivity == pytest.approx(expected, abs=1e-6)


class TestRefusals:
    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            SpectralBisection(Graph(directed=True))

    def test_a_disconnected_graph_is_refused(self):
        g = _path(3)
        g.add_node("island")
        with pytest.raises(Invalid):
            SpectralBisection(g)

    def test_a_single_node_is_refused(self):
        g = Graph()
        g.add_node("a")
        with pytest.raises(Invalid):
            SpectralBisection(g)


class TestReport:
    def test_the_note_states_the_cut_fraction(self):
        note = SpectralBisection(_two_cliques_bridged()).note()
        assert "cuts 1 of 13 edge(s)" in note
