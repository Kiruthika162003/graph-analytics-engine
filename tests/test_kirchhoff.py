from __future__ import annotations

import random
from itertools import combinations

import pytest

from mesh.errors import Invalid
from mesh.graph import Graph
from mesh.kirchhoff import SpanningTreeCount
from mesh.unionfind import UnionFind


def _complete(k: int) -> Graph:
    g = Graph()
    nodes = [str(i) for i in range(k)]
    for n in nodes:
        g.add_node(n)
    for a, b in combinations(nodes, 2):
        g.add_edge(a, b)
    return g


def _brute_count(g: Graph) -> int:
    # every edge subset of size n-1 that connects all nodes is a spanning tree
    nodes = g.nodes()
    edges = g.edges()
    total = 0
    for subset in combinations(edges, len(nodes) - 1):
        uf = UnionFind()
        for n in nodes:
            uf.add(n)
        if all(uf.union(u, v) for u, v, _w in subset):
            total += 1
    return total


class TestCount:
    def test_a_tree_has_exactly_one_spanning_tree(self):
        g = Graph()
        for n in "abcd":
            g.add_node(n)
        for u, v in [("a", "b"), ("b", "c"), ("c", "d")]:
            g.add_edge(u, v)
        assert SpanningTreeCount(g).count == 1

    def test_a_cycle_of_k_has_k_spanning_trees(self):
        g = Graph()
        for n in "abcde":
            g.add_node(n)
        for i, n in enumerate("abcde"):
            g.add_edge(n, "abcde"[(i + 1) % 5])
        assert SpanningTreeCount(g).count == 5

    def test_a_complete_graph_follows_cayleys_formula(self):
        # K5 has 5**3 = 125 spanning trees
        assert SpanningTreeCount(_complete(5)).count == 125

    def test_a_disconnected_graph_has_zero_spanning_trees(self):
        g = Graph()
        for n in "abcd":
            g.add_node(n)
        g.add_edge("a", "b")
        g.add_edge("c", "d")
        assert SpanningTreeCount(g).count == 0

    def test_a_single_node_counts_as_one(self):
        g = Graph()
        g.add_node("solo")
        assert SpanningTreeCount(g).count == 1


class TestCrossCheck:
    def test_two_deleted_rows_agree(self):
        stc = SpanningTreeCount(_complete(4))
        assert stc.count == stc.cross_check == 16


class TestRefusals:
    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            SpanningTreeCount(Graph(directed=True))

    def test_an_empty_graph_is_refused(self):
        with pytest.raises(Invalid):
            SpanningTreeCount(Graph())


class TestAgainstBruteForce:
    def test_the_determinant_matches_enumerating_edge_subsets(self):
        rng = random.Random(113)
        for _ in range(25):
            g = Graph()
            nodes = [str(i) for i in range(6)]
            for n in nodes:
                g.add_node(n)
            for a, b in combinations(nodes, 2):
                if rng.random() < 0.5:
                    g.add_edge(a, b)
            assert SpanningTreeCount(g).count == _brute_count(g)


class TestReport:
    def test_the_note_states_count_and_maximum(self):
        note = SpanningTreeCount(_complete(4)).note()
        assert "16 spanning tree(s)" in note
        assert "maximum of 16" in note
