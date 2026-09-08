from __future__ import annotations

import random
from itertools import combinations

import pytest

from mesh.bronkerbosch import BronKerbosch
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


def _brute_maximal_cliques(g: Graph) -> set[frozenset[str]]:
    nodes = g.nodes()
    is_clique = {}
    for size in range(1, len(nodes) + 1):
        for combo in combinations(nodes, size):
            ok = all(g.has_edge(a, b) for a, b in combinations(combo, 2))
            is_clique[frozenset(combo)] = ok
    cliques = {c for c, ok in is_clique.items() if ok}
    return {c for c in cliques if not any(c < d for d in cliques)}


class TestCliques:
    def test_a_complete_graph_is_one_maximal_clique(self):
        bk = BronKerbosch(_complete(4))
        assert bk.cliques == [frozenset("0123")]
        assert bk.clique_number() == 4

    def test_two_triangles_sharing_a_node_are_two_cliques(self):
        g = Graph()
        for n in "abcde":
            g.add_node(n)
        for x, y in [("a", "b"), ("b", "c"), ("c", "a"), ("c", "d"), ("d", "e"), ("e", "c")]:
            g.add_edge(x, y)
        bk = BronKerbosch(g)
        assert set(bk.cliques) == {frozenset("abc"), frozenset("cde")}

    def test_a_star_yields_one_clique_per_edge(self):
        g = Graph()
        g.add_node("hub")
        for leaf in "abc":
            g.add_node(leaf)
            g.add_edge("hub", leaf)
        bk = BronKerbosch(g)
        assert len(bk.cliques) == 3
        assert bk.clique_number() == 2

    def test_the_largest_comes_first(self):
        g = _complete(3)
        g.add_node("tail")
        g.add_edge("0", "tail")
        assert BronKerbosch(g).largest() == frozenset("012")

    def test_an_isolated_node_is_its_own_clique(self):
        g = Graph()
        g.add_node("solo")
        assert BronKerbosch(g).cliques == [frozenset({"solo"})]


class TestPivot:
    def test_pivoting_branches_less_than_the_node_count_on_a_clique(self):
        # on K6 the pivot leaves nothing to branch on beyond a single path
        bk = BronKerbosch(_complete(6))
        assert bk.branches <= 6


class TestRefusals:
    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            BronKerbosch(Graph(directed=True))

    def test_the_largest_of_an_empty_graph_is_refused(self):
        with pytest.raises(Invalid):
            BronKerbosch(Graph()).largest()


class TestAgainstBruteForce:
    def test_it_matches_checking_every_subset(self):
        rng = random.Random(89)
        for _ in range(30):
            g = Graph()
            nodes = [str(i) for i in range(8)]
            for n in nodes:
                g.add_node(n)
            for a, b in combinations(nodes, 2):
                if rng.random() < 0.45:
                    g.add_edge(a, b)
            assert set(BronKerbosch(g).cliques) == _brute_maximal_cliques(g)


class TestReport:
    def test_the_note_states_count_and_clique_number(self):
        note = BronKerbosch(_complete(3)).note()
        assert "1 maximal clique(s)" in note
        assert "clique number 3" in note
