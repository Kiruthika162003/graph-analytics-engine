from __future__ import annotations

import pytest

from mesh.closeness import Closeness
from mesh.errors import Invalid
from mesh.graph import Graph


def _path5() -> Graph:
    # a - b - c - d - e: c is the center
    g = Graph()
    for n in "abcde":
        g.add_node(n)
    for u, v in [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e")]:
        g.add_edge(u, v)
    return g


class TestClassic:
    def test_the_center_of_a_path_is_closest(self):
        assert Closeness(_path5()).top(1, harmonic=False)[0][0] == "c"

    def test_classic_is_the_reciprocal_of_the_distance_sum(self):
        # c is at distances 2,1,1,2 to the others: sum 6, reciprocal 4/6
        c = Closeness(_path5())
        assert c.classic["c"] == pytest.approx(4 / 6)

    def test_an_endpoint_scores_lower_than_the_center(self):
        c = Closeness(_path5())
        assert c.classic["a"] < c.classic["c"]


class TestHarmonic:
    def test_harmonic_sums_reciprocal_distances(self):
        # c: 1/2 + 1/1 + 1/1 + 1/2 = 3, over 4 others
        c = Closeness(_path5())
        assert c.harmonic["c"] == pytest.approx(3 / 4)

    def test_harmonic_and_classic_agree_on_a_connected_path(self):
        c = Closeness(_path5())
        assert c.top(1, harmonic=True)[0][0] == c.top(1, harmonic=False)[0][0]

    def test_harmonic_stays_finite_and_ranked_when_disconnected(self):
        g = _path5()
        g.add_node("island")
        c = Closeness(g)
        assert c.disconnected
        assert c.harmonic["island"] == 0.0
        assert c.harmonic["c"] > c.harmonic["a"] > 0.0

    def test_unreachable_nodes_contribute_nothing_to_harmonic(self):
        # two disjoint edges: each node reaches one other at distance 1
        g = Graph()
        for n in "abcd":
            g.add_node(n)
        g.add_edge("a", "b")
        g.add_edge("c", "d")
        c = Closeness(g)
        assert c.harmonic["a"] == pytest.approx(1 / 3)


class TestRefusal:
    def test_a_single_node_is_refused(self):
        g = Graph()
        g.add_node("a")
        with pytest.raises(Invalid):
            Closeness(g)


class TestReport:
    def test_the_note_names_the_disconnection(self):
        g = _path5()
        g.add_node("island")
        assert "disconnected" in Closeness(g).note()

    def test_the_note_names_the_center_when_connected(self):
        assert "'c'" in Closeness(_path5()).note()
