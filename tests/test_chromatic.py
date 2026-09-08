from __future__ import annotations

import random
from itertools import combinations

import pytest

from mesh.chromatic import ChromaticNumber
from mesh.errors import Invalid
from mesh.graph import Graph


def _cycle(k: int) -> Graph:
    g = Graph()
    nodes = [str(i) for i in range(k)]
    for n in nodes:
        g.add_node(n)
    for i in range(k):
        g.add_edge(nodes[i], nodes[(i + 1) % k])
    return g


def _brute_chromatic(g: Graph) -> int:
    nodes = g.nodes()
    edges = [(u, v) for u, v, _w in g.edges()]
    for k in range(1, len(nodes) + 1):
        for mask in range(k ** len(nodes)):
            colors = {}
            m = mask
            for n in nodes:
                colors[n] = m % k
                m //= k
            if all(colors[u] != colors[v] for u, v in edges):
                return k
    return len(nodes)


class TestValue:
    def test_an_odd_cycle_needs_three(self):
        assert ChromaticNumber(_cycle(5)).value == 3

    def test_an_even_cycle_needs_two(self):
        assert ChromaticNumber(_cycle(6)).value == 2

    def test_a_complete_graph_needs_every_color(self):
        g = Graph()
        for n in "abcde":
            g.add_node(n)
        for a, b in combinations("abcde", 2):
            g.add_edge(a, b)
        cn = ChromaticNumber(g)
        assert cn.value == 5
        assert "the bounds met" in cn.note()

    def test_the_coloring_achieves_the_value_and_is_proper(self):
        cn = ChromaticNumber(_cycle(7))
        assert cn.is_proper()
        assert len(set(cn.coloring.values())) == cn.value

    def test_an_edgeless_graph_needs_one(self):
        g = Graph()
        for n in "abc":
            g.add_node(n)
        assert ChromaticNumber(g).value == 1


class TestBounds:
    def test_the_value_sits_between_clique_and_greedy(self):
        rng = random.Random(293)
        for _ in range(20):
            g = Graph()
            nodes = [str(i) for i in range(9)]
            for n in nodes:
                g.add_node(n)
            for a, b in combinations(nodes, 2):
                if rng.random() < 0.4:
                    g.add_edge(a, b)
            cn = ChromaticNumber(g)
            assert cn.lower <= cn.value <= cn.upper
            assert cn.greedy_gap() >= 0


class TestRefusals:
    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            ChromaticNumber(Graph(directed=True))

    def test_an_empty_graph_is_refused(self):
        with pytest.raises(Invalid):
            ChromaticNumber(Graph())

    def test_a_graph_over_the_cap_is_refused(self):
        g = Graph()
        for i in range(41):
            g.add_node(str(i))
        with pytest.raises(Invalid):
            ChromaticNumber(g)


class TestAgainstBruteForce:
    def test_the_value_matches_trying_every_coloring(self):
        rng = random.Random(307)
        for _ in range(20):
            g = Graph()
            nodes = [str(i) for i in range(6)]
            for n in nodes:
                g.add_node(n)
            for a, b in combinations(nodes, 2):
                if rng.random() < 0.5:
                    g.add_edge(a, b)
            assert ChromaticNumber(g).value == _brute_chromatic(g)


class TestReport:
    def test_the_note_states_the_greedy_gap(self):
        note = ChromaticNumber(_cycle(5)).note()
        assert "chromatic number 3" in note
        assert "gap of" in note
