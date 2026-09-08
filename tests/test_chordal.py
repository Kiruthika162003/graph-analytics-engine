from __future__ import annotations

import random
from itertools import combinations

import pytest

from mesh.chordal import Chordal
from mesh.chromatic import ChromaticNumber
from mesh.errors import Invalid
from mesh.factories import complete, cycle, path, star, wheel
from mesh.girth import Girth
from mesh.graph import Graph


def _fan(n: int) -> Graph:
    # a path with a hub joined to every node; every cycle through the hub has a chord
    g = path(n)
    g.add_node("hub")
    for i in range(n):
        g.add_edge("hub", str(i))
    return g


def _brute_chordal(g: Graph) -> bool:
    # a chordless cycle of length four or more is the one obstruction
    nodes = g.nodes()
    for size in range(4, len(nodes) + 1):
        for subset in combinations(nodes, size):
            sub = Graph()
            for n in subset:
                sub.add_node(n)
            for a, b in combinations(subset, 2):
                if g.has_edge(a, b):
                    sub.add_edge(a, b)
            # an induced cycle: every node degree two, connected, girth = size
            if all(sub.degree(n) == 2 for n in subset) and Girth(sub).girth == size:
                return False
    return True


class TestVerdict:
    def test_trees_and_complete_graphs_are_chordal(self):
        assert Chordal(path(6)).is_chordal
        assert Chordal(star(5)).is_chordal
        assert Chordal(complete(5)).is_chordal

    def test_a_square_is_not_chordal_and_names_a_witness(self):
        c = Chordal(cycle(4))
        assert not c.is_chordal
        assert c.witness in cycle(4).nodes()

    def test_a_triangle_is_chordal_but_a_pentagon_is_not(self):
        assert Chordal(cycle(3)).is_chordal
        assert not Chordal(cycle(5)).is_chordal

    def test_a_fan_is_chordal_but_a_wheel_is_not(self):
        # the guess was that the hub chords every rim cycle of a wheel; the hub is not
        # on the rim, so the rim itself is a chordless hexagon and the wheel fails.
        # a fan, a path with a hub, has no cycle that avoids the hub, and is chordal
        assert Chordal(_fan(6)).is_chordal
        w = Chordal(wheel(6))
        assert not w.is_chordal
        assert w.witness != "hub"


class TestExactNumbers:
    def test_clique_number_matches_bron_kerbosch_on_chordal_graphs(self):
        for g in (path(5), star(4), complete(6), _fan(5)):
            c = Chordal(g)
            assert c.matches_bron_kerbosch()

    def test_chromatic_number_equals_the_clique_number(self):
        for g in (star(4), _fan(5), complete(4)):
            c = Chordal(g)
            assert c.chromatic_number() == ChromaticNumber(g).value

    def test_treewidth_of_a_tree_is_one(self):
        assert Chordal(path(7)).treewidth() == 1

    def test_the_clique_number_is_refused_off_a_chordal_graph(self):
        with pytest.raises(Invalid):
            Chordal(cycle(4)).clique_number()


class TestAgainstBruteForce:
    def test_the_verdict_matches_searching_for_a_chordless_cycle(self):
        rng = random.Random(443)
        for _ in range(30):
            g = Graph()
            nodes = [str(i) for i in range(7)]
            for n in nodes:
                g.add_node(n)
            for a, b in combinations(nodes, 2):
                if rng.random() < 0.45:
                    g.add_edge(a, b)
            assert Chordal(g).is_chordal == _brute_chordal(g)


class TestRefusal:
    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            Chordal(Graph(directed=True))


class TestReport:
    def test_the_note_states_the_exact_numbers(self):
        note = Chordal(_fan(5)).note()
        assert "chordal: clique number 3" in note
        assert "treewidth 2" in note

    def test_the_note_names_the_witness_when_not_chordal(self):
        assert "not chordal" in Chordal(cycle(4)).note()
