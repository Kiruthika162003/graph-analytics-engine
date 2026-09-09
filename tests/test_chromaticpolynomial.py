from __future__ import annotations

import random
from itertools import combinations, product

import pytest

from mesh.chromatic import ChromaticNumber
from mesh.chromaticpolynomial import ChromaticPolynomial
from mesh.errors import Invalid
from mesh.factories import complete, cycle, path, star
from mesh.graph import Graph


def _brute_count(g: Graph, k: int) -> int:
    nodes = g.nodes()
    edges = [(u, v) for u, v, _w in g.edges()]
    total = 0
    for colors in product(range(k), repeat=len(nodes)):
        c = dict(zip(nodes, colors, strict=True))
        if all(c[u] != c[v] for u, v in edges):
            total += 1
    return total


class TestClosedForms:
    def test_a_tree_gives_k_times_k_minus_one_to_the_n_minus_one(self):
        for g in (path(5), star(4)):
            cp = ChromaticPolynomial(g)
            n = g.node_count()
            for k in range(0, 5):
                assert cp.evaluate(k) == k * (k - 1) ** (n - 1)

    def test_a_cycle_gives_the_alternating_form(self):
        for n in (3, 4, 5, 6):
            cp = ChromaticPolynomial(cycle(n))
            for k in range(0, 5):
                assert cp.evaluate(k) == (k - 1) ** n + (-1) ** n * (k - 1)

    def test_a_complete_graph_gives_the_falling_factorial(self):
        cp = ChromaticPolynomial(complete(4))
        for k in range(0, 7):
            assert cp.evaluate(k) == k * (k - 1) * (k - 2) * (k - 3)

    def test_an_edgeless_graph_gives_k_to_the_n(self):
        g = Graph()
        for n in "abc":
            g.add_node(n)
        assert ChromaticPolynomial(g).coefficients == (0, 0, 0, 1)
        assert ChromaticPolynomial(Graph()).coefficients == (1,)


class TestAgainstBruteForce:
    def test_evaluations_match_counting_colorings_on_random_graphs(self):
        rng = random.Random(643)
        for _ in range(12):
            g = Graph()
            nodes = [str(i) for i in range(6)]
            for n in nodes:
                g.add_node(n)
            for a, b in combinations(nodes, 2):
                if rng.random() < 0.4:
                    g.add_edge(a, b)
            cp = ChromaticPolynomial(g)
            for k in range(0, 4):
                assert cp.evaluate(k) == _brute_count(g, k)
            assert cp.leading_terms_hold()
            assert cp.chromatic_number() == ChromaticNumber(g).value


class TestShape:
    def test_the_leading_terms_are_one_and_minus_the_edge_count(self):
        cp = ChromaticPolynomial(cycle(5))
        assert cp.degree() == 5
        assert cp.coefficients[-1] == 1
        assert cp.coefficients[-2] == -5
        assert cp.leading_terms_hold()

    def test_the_memo_keeps_the_call_count_small(self):
        cp = ChromaticPolynomial(complete(5))
        assert cp.calls < 200


class TestRefusal:
    def test_directed_and_oversized_graphs_are_refused(self):
        with pytest.raises(Invalid):
            ChromaticPolynomial(Graph(directed=True))
        with pytest.raises(Invalid):
            ChromaticPolynomial(path(13))


class TestReport:
    def test_the_note_prints_the_polynomial_and_the_number(self):
        note = ChromaticPolynomial(cycle(3)).note()
        assert "P(k) = 1k^3 + -3k^2 + 2k^1" in note
        assert "chromatic number 3" in note
