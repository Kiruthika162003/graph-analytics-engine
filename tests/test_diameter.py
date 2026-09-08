from __future__ import annotations

import random
from itertools import combinations, pairwise

import pytest

from mesh.diameter import Diameter
from mesh.errors import Invalid
from mesh.graph import Graph


def _path(k: int) -> Graph:
    g = Graph()
    nodes = [str(i) for i in range(k)]
    for n in nodes:
        g.add_node(n)
    for a, b in pairwise(nodes):
        g.add_edge(a, b)
    return g


class TestMeasures:
    def test_a_path_of_five_has_diameter_four(self):
        assert Diameter(_path(5)).diameter() == 4

    def test_the_center_of_an_odd_path_is_its_middle(self):
        d = Diameter(_path(5))
        assert d.center() == {"2"}
        assert d.radius() == 2

    def test_the_periphery_of_a_path_is_its_two_ends(self):
        assert Diameter(_path(5)).periphery() == {"0", "4"}

    def test_a_complete_graph_has_diameter_one(self):
        g = Graph()
        for n in "abcd":
            g.add_node(n)
        for a, b in combinations("abcd", 2):
            g.add_edge(a, b)
        d = Diameter(g)
        assert d.diameter() == 1
        assert d.center() == {"a", "b", "c", "d"}

    def test_a_single_node_has_diameter_zero(self):
        g = Graph()
        g.add_node("solo")
        assert Diameter(g).diameter() == 0


class TestDoubleSweep:
    def test_the_sweep_is_exact_on_a_tree(self):
        rng = random.Random(5)
        for _ in range(25):
            n = rng.randint(2, 25)
            g = Graph()
            g.add_node("0")
            for i in range(1, n):
                g.add_node(str(i))
                g.add_edge(str(rng.randrange(i)), str(i))
            d = Diameter(g)
            assert d.double_sweep() == d.diameter()

    def test_the_sweep_never_overestimates(self):
        rng = random.Random(6)
        for _ in range(30):
            g = _path(6)
            for a, b in combinations(g.nodes(), 2):
                if not g.has_edge(a, b) and rng.random() < 0.25:
                    g.add_edge(a, b)
            d = Diameter(g)
            assert d.double_sweep() <= d.diameter()

    def test_the_sweep_can_undershoot_on_a_general_graph(self):
        # first guess: the sweep is always exact. A 5-cycle with one pendant
        # did not refute it, the sweep found the pendant and read the true
        # diameter. This shape does: a hexagon a-f with pendants "0" on b
        # and "1" on e, started from d. The farthest nodes from d are the
        # cycle vertex a and the pendant "0", both at 3; the tie goes to a,
        # whose own eccentricity is 3, while the true diameter "0" to "1"
        # is 5. The sweep reads 3 and is short by 2.
        g = Graph()
        for n in ["a", "b", "c", "d", "e", "f", "0", "1"]:
            g.add_node(n)
        for u, v in [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("e", "f"), ("f", "a")]:
            g.add_edge(u, v)
        g.add_edge("b", "0")
        g.add_edge("e", "1")
        d = Diameter(g)
        assert d.diameter() == 5
        assert d.double_sweep(start="d") == 3


class TestRefusals:
    def test_a_disconnected_graph_is_refused(self):
        g = _path(3)
        g.add_node("island")
        with pytest.raises(Invalid) as caught:
            Diameter(g)
        assert "infinite" in str(caught.value)

    def test_an_empty_graph_is_refused(self):
        with pytest.raises(Invalid):
            Diameter(Graph())


class TestReport:
    def test_the_note_states_diameter_and_center(self):
        note = Diameter(_path(5)).note()
        assert "diameter 4" in note
        assert "center ['2']" in note
