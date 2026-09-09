from __future__ import annotations

import random
from itertools import combinations
from math import cos, pi

import pytest

from mesh.errors import Invalid
from mesh.factories import complete, cycle, path, star
from mesh.graph import Graph
from mesh.laplacianspectrum import LaplacianSpectrum


def _two_cliques() -> Graph:
    g = Graph()
    for group in ("abcd", "wxyz"):
        for n in group:
            g.add_node(n)
        for a, b in combinations(group, 2):
            g.add_edge(a, b)
    g.add_edge("d", "w")
    return g


class TestClosedForms:
    def test_a_complete_graph_has_zero_then_n_repeated(self):
        ls = LaplacianSpectrum(complete(5))
        assert ls.values == pytest.approx([0, 5, 5, 5, 5])

    def test_a_path_follows_the_cosine_form(self):
        n = 6
        expected = sorted(2 - 2 * cos(pi * k / n) for k in range(n))
        assert LaplacianSpectrum(path(n)).values == pytest.approx(expected, abs=1e-9)

    def test_a_star_has_zero_ones_and_n(self):
        ls = LaplacianSpectrum(star(4))
        assert ls.values == pytest.approx([0, 1, 1, 1, 5])


class TestConnectivity:
    def test_zeros_count_components(self):
        g = Graph()
        for n in "abcdef":
            g.add_node(n)
        g.add_edge("a", "b")
        g.add_edge("c", "d")
        ls = LaplacianSpectrum(g)
        assert ls.components() == 4
        assert not ls.is_connected()
        assert ls.spanning_trees() == 0.0

    def test_algebraic_connectivity_is_positive_when_connected_and_rises_with_density(self):
        sparse = LaplacianSpectrum(path(6)).algebraic_connectivity()
        dense = LaplacianSpectrum(complete(6)).algebraic_connectivity()
        assert 0 < sparse < dense
        assert LaplacianSpectrum(cycle(6)).is_connected()

    def test_the_fiedler_vector_separates_two_cliques_at_the_bridge(self):
        ls = LaplacianSpectrum(_two_cliques())
        left, right = ls.fiedler_partition()
        assert {tuple(left), tuple(right)} == {tuple("abcd"), tuple("wxyz")}
        assert ls.cut_size() == 1


class TestIdentities:
    def test_trace_and_tree_count_agree_on_random_graphs(self):
        rng = random.Random(709)
        for _ in range(12):
            g = Graph()
            nodes = [str(i) for i in range(7)]
            for n in nodes:
                g.add_node(n)
            for a, b in combinations(nodes, 2):
                if rng.random() < 0.5:
                    g.add_edge(a, b)
            ls = LaplacianSpectrum(g)
            assert ls.trace_identity_holds()
            assert ls.agrees_with_kirchhoff()

    def test_a_cycle_has_n_spanning_trees_from_its_spectrum(self):
        assert LaplacianSpectrum(cycle(7)).spanning_trees() == pytest.approx(7.0)


class TestRefusal:
    def test_a_directed_graph_is_refused_and_an_empty_one_reads_zero(self):
        with pytest.raises(Invalid):
            LaplacianSpectrum(Graph(directed=True))
        ls = LaplacianSpectrum(Graph())
        assert ls.components() == 0
        assert ls.agrees_with_kirchhoff()


class TestReport:
    def test_the_note_reads_components_connectivity_trees_and_the_cut(self):
        # the guess was 32 trees; each K4 has 16 and the bridge must be in every tree,
        # so the product is 256
        note = LaplacianSpectrum(_two_cliques()).note()
        assert "1 component(s)" in note
        assert "256 spanning tree(s)" in note
        assert "Fiedler cut of 1 edge(s) splits 4 from 4" in note
