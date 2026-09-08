from __future__ import annotations

import random
from itertools import combinations, permutations

import pytest

from mesh.errors import Invalid
from mesh.graph import Graph
from mesh.hamiltonian import Hamiltonian


def _cycle(k: int) -> Graph:
    g = Graph()
    nodes = [str(i) for i in range(k)]
    for n in nodes:
        g.add_node(n)
    for i in range(k):
        g.add_edge(nodes[i], nodes[(i + 1) % k])
    return g


def _brute_path(g: Graph, cycle: bool) -> bool:
    nodes = g.nodes()
    for perm in permutations(nodes):
        ok = all(g.has_edge(perm[i], perm[i + 1]) for i in range(len(perm) - 1))
        if ok and (not cycle or g.has_edge(perm[-1], perm[0])):
            return True
    return False


class TestExistence:
    def test_a_cycle_graph_has_a_hamiltonian_cycle(self):
        h = Hamiltonian(_cycle(6), cycle=True)
        assert h.exists
        assert h.witness is not None
        assert len(h.witness) == 6

    def test_a_path_graph_has_a_path_but_no_cycle(self):
        g = Graph()
        for n in "abcd":
            g.add_node(n)
        for u, v in [("a", "b"), ("b", "c"), ("c", "d")]:
            g.add_edge(u, v)
        assert Hamiltonian(g).exists
        h = Hamiltonian(g, cycle=True)
        assert not h.exists
        assert "degree below two" in h.reason

    def test_a_star_has_no_hamiltonian_path(self):
        g = Graph()
        g.add_node("hub")
        for leaf in "abc":
            g.add_node(leaf)
            g.add_edge("hub", leaf)
        h = Hamiltonian(g)
        assert not h.exists
        assert "more than two nodes of degree one" in h.reason

    def test_the_witness_is_a_real_walk_visiting_everyone_once(self):
        g = _cycle(7)
        g.add_edge("0", "3")
        h = Hamiltonian(g, cycle=True)
        w = h.witness
        assert sorted(w) == sorted(g.nodes())
        for i in range(len(w)):
            assert g.has_edge(w[i], w[(i + 1) % len(w)])

    def test_dirac_applies_to_a_dense_graph(self):
        g = Graph()
        for n in "abcdef":
            g.add_node(n)
        for a, b in combinations("abcdef", 2):
            if {a, b} not in ({"a", "b"}, {"c", "d"}, {"e", "f"}):
                g.add_edge(a, b)  # every degree is 4, half of 6 is 3
        h = Hamiltonian(g, cycle=True)
        assert h.dirac
        assert h.exists
        assert "Dirac guaranteed it" in h.note()

    def test_a_disconnected_graph_is_ruled_out(self):
        g = _cycle(3)
        g.add_node("island")
        h = Hamiltonian(g)
        assert not h.exists
        assert h.reason == "disconnected"


class TestRefusals:
    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            Hamiltonian(Graph(directed=True))

    def test_a_graph_over_the_cap_is_refused(self):
        with pytest.raises(Invalid):
            Hamiltonian(_cycle(15))


class TestAgainstBruteForce:
    def test_the_verdict_matches_trying_every_ordering(self):
        rng = random.Random(317)
        for _ in range(25):
            g = Graph()
            nodes = [str(i) for i in range(6)]
            for n in nodes:
                g.add_node(n)
            for a, b in combinations(nodes, 2):
                if rng.random() < 0.45:
                    g.add_edge(a, b)
            for cycle in (False, True):
                assert Hamiltonian(g, cycle=cycle).exists == _brute_path(g, cycle)


class TestReport:
    def test_the_note_counts_branches_against_the_bound(self):
        note = Hamiltonian(_cycle(5), cycle=True).note()
        assert "unpruned bound of 24" in note
