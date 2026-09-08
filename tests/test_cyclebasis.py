from __future__ import annotations

import random
from itertools import combinations

import pytest

from mesh.cyclebasis import CycleBasis
from mesh.errors import Invalid
from mesh.graph import Graph


def _square_with_diagonal() -> Graph:
    g = Graph()
    for n in "abcd":
        g.add_node(n)
    for u, v in [("a", "b"), ("b", "c"), ("c", "d"), ("d", "a"), ("a", "c")]:
        g.add_edge(u, v)
    return g


class TestBasis:
    def test_a_tree_has_no_cycles(self):
        g = Graph()
        for n in "abc":
            g.add_node(n)
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        cb = CycleBasis(g)
        assert cb.cycles == []
        assert cb.cyclomatic_number() == 0

    def test_a_square_with_a_diagonal_has_two_independent_cycles(self):
        cb = CycleBasis(_square_with_diagonal())
        assert cb.cyclomatic_number() == 2
        assert len(cb.cycles) == 2

    def test_every_basis_cycle_is_a_real_closed_walk(self):
        cb = CycleBasis(_square_with_diagonal())
        for cycle in cb.cycles:
            assert cb.is_closed_walk(cycle)

    def test_each_cycle_contains_its_own_chord(self):
        cb = CycleBasis(_square_with_diagonal())
        for (u, v), cycle in zip(cb.chords, cb.cycles, strict=True):
            assert u in cycle and v in cycle

    def test_the_count_is_edges_minus_nodes_plus_components(self):
        g = _square_with_diagonal()
        g.add_node("x")
        g.add_node("y")
        g.add_edge("x", "y")
        cb = CycleBasis(g)
        assert cb.components == 2
        assert cb.cyclomatic_number() == 6 - 6 + 2
        assert len(cb.cycles) == cb.cyclomatic_number()


class TestRefusal:
    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            CycleBasis(Graph(directed=True))


class TestAgainstFormula:
    def test_basis_size_and_validity_on_random_graphs(self):
        rng = random.Random(163)
        for _ in range(30):
            g = Graph()
            nodes = [str(i) for i in range(8)]
            for n in nodes:
                g.add_node(n)
            for a, b in combinations(nodes, 2):
                if rng.random() < 0.35:
                    g.add_edge(a, b)
            cb = CycleBasis(g)
            assert len(cb.cycles) == cb.cyclomatic_number()
            for cycle in cb.cycles:
                assert cb.is_closed_walk(cycle)
                assert len(cycle) == len(set(cycle))  # a simple cycle, no repeats


class TestReport:
    def test_the_note_states_the_chord_share(self):
        note = CycleBasis(_square_with_diagonal()).note()
        assert "2 independent cycle(s)" in note
        assert "2 chord(s) among 5 edge(s)" in note
