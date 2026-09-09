from __future__ import annotations

import random
from itertools import combinations

import pytest

from mesh.errors import Invalid
from mesh.factories import complete, cycle, path, star
from mesh.graph import Graph
from mesh.independencepolynomial import IndependencePolynomial


def _brute(g: Graph) -> list[int]:
    nodes = g.nodes()
    counts = [0] * (len(nodes) + 1)
    for k in range(len(nodes) + 1):
        for subset in combinations(nodes, k):
            if not any(g.has_edge(a, b) for a, b in combinations(subset, 2)):
                counts[k] += 1
    while len(counts) > 1 and counts[-1] == 0:
        counts.pop()
    return counts


class TestClosedForms:
    def test_an_edgeless_graph_gives_the_binomial_row(self):
        g = Graph()
        for n in "abcd":
            g.add_node(n)
        assert IndependencePolynomial(g).coefficients == (1, 4, 6, 4, 1)

    def test_a_complete_graph_gives_one_plus_n_x(self):
        assert IndependencePolynomial(complete(5)).coefficients == (1, 5)

    def test_a_star_takes_the_hub_alone_or_any_leaves(self):
        # leaves form 2^4 sets, plus the hub by itself
        ip = IndependencePolynomial(star(4))
        assert ip.total_sets() == 16 + 1
        assert ip.independence_number() == 4

    def test_a_path_sums_to_fibonacci(self):
        fib = [1, 2, 3, 5, 8, 13, 21, 34]
        for n in range(1, 8):
            assert IndependencePolynomial(path(n)).total_sets() == fib[n]

    def test_the_empty_graph_has_the_empty_set_only(self):
        assert IndependencePolynomial(Graph()).coefficients == (1,)


class TestAgainstBruteForce:
    def test_coefficients_match_enumeration_on_random_graphs(self):
        rng = random.Random(661)
        for _ in range(15):
            g = Graph()
            nodes = [str(i) for i in range(8)]
            for n in nodes:
                g.add_node(n)
            for a, b in combinations(nodes, 2):
                if rng.random() < 0.35:
                    g.add_edge(a, b)
            ip = IndependencePolynomial(g)
            assert list(ip.coefficients) == _brute(g)
            assert ip.matches_complement_clique()

    def test_the_second_coefficient_counts_non_edges(self):
        g = cycle(6)
        assert IndependencePolynomial(g).coefficients[2] == 15 - 6


class TestMatchings:
    def test_the_matching_polynomial_of_a_path_counts_edge_subsets(self):
        # path of four nodes has three edges; matchings: empty, three singles, one pair
        assert IndependencePolynomial(path(4)).matching_polynomial() == (1, 3, 1)

    def test_a_triangle_has_three_matchings_of_one_edge_and_none_larger(self):
        assert IndependencePolynomial(cycle(3)).matching_polynomial() == (1, 3)


class TestRefusal:
    def test_directed_and_oversized_graphs_are_refused(self):
        with pytest.raises(Invalid):
            IndependencePolynomial(Graph(directed=True))
        with pytest.raises(Invalid):
            IndependencePolynomial(path(21))


class TestReport:
    def test_the_note_lists_coefficients_and_totals(self):
        note = IndependencePolynomial(cycle(4)).note()
        assert "I(x) coefficients [1, 4, 2]" in note
        assert "7 independent set(s), independence number 2" in note
