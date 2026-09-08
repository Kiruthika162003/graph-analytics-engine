from __future__ import annotations

import random
from itertools import permutations

import pytest

from mesh.errors import Invalid
from mesh.graph import Graph
from mesh.toposortcount import TopologicalCount


def _brute_count(g: Graph) -> int:
    nodes = g.nodes()
    edges = [(u, v) for u, v, _w in g.edges()]
    total = 0
    for perm in permutations(nodes):
        rank = {n: i for i, n in enumerate(perm)}
        if all(rank[u] < rank[v] for u, v in edges):
            total += 1
    return total


class TestCount:
    def test_a_chain_has_exactly_one_order(self):
        g = Graph(directed=True)
        for n in "abcd":
            g.add_node(n)
        for u, v in [("a", "b"), ("b", "c"), ("c", "d")]:
            g.add_edge(u, v)
        assert TopologicalCount(g).count == 1

    def test_independent_tasks_have_every_permutation(self):
        g = Graph(directed=True)
        for n in "abcd":
            g.add_node(n)
        tc = TopologicalCount(g)
        assert tc.count == 24
        assert tc.freedom() == 1.0

    def test_a_diamond_has_two_orders(self):
        g = Graph(directed=True)
        for n in "abcd":
            g.add_node(n)
        for u, v in [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")]:
            g.add_edge(u, v)
        assert TopologicalCount(g).count == 2

    def test_a_cycle_has_zero_orders(self):
        g = Graph(directed=True)
        for n in "abc":
            g.add_node(n)
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        g.add_edge("c", "a")
        tc = TopologicalCount(g)
        assert tc.count == 0
        assert not tc.is_acyclic()
        assert "cycle" in tc.note()

    def test_an_empty_graph_has_one_order(self):
        assert TopologicalCount(Graph(directed=True)).count == 1


class TestRefusals:
    def test_an_undirected_graph_is_refused(self):
        with pytest.raises(Invalid):
            TopologicalCount(Graph())

    def test_a_graph_over_the_cap_is_refused(self):
        g = Graph(directed=True)
        for i in range(21):
            g.add_node(str(i))
        with pytest.raises(Invalid):
            TopologicalCount(g)


class TestAgainstBruteForce:
    def test_the_count_matches_checking_every_permutation(self):
        rng = random.Random(349)
        for _ in range(30):
            g = Graph(directed=True)
            nodes = [str(i) for i in range(6)]
            for n in nodes:
                g.add_node(n)
            for i, u in enumerate(nodes):
                for v in nodes[i + 1 :]:
                    if rng.random() < 0.3:
                        g.add_edge(u, v)
            assert TopologicalCount(g).count == _brute_count(g)


class TestReport:
    def test_the_note_states_the_freedom(self):
        g = Graph(directed=True)
        for n in "abc":
            g.add_node(n)
        g.add_edge("a", "b")
        note = TopologicalCount(g).note()
        assert "3 topological order(s), 50.0% of the 6 permutations" in note
