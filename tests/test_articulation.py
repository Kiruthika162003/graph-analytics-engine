from __future__ import annotations

import random

import pytest

from mesh.articulation import Articulation
from mesh.connectedcomponents import ConnectedComponents
from mesh.errors import Invalid
from mesh.graph import Graph


def _without_node(g: Graph, gone: str) -> Graph:
    h = Graph()
    for n in g.nodes():
        if n != gone:
            h.add_node(n)
    for u, v, w in g.edges():
        if gone not in (u, v):
            h.add_edge(u, v, w)
    return h


def _without_edge(g: Graph, a: str, b: str) -> Graph:
    h = Graph()
    for n in g.nodes():
        h.add_node(n)
    for u, v, w in g.edges():
        if {u, v} != {a, b}:
            h.add_edge(u, v, w)
    return h


def _two_triangles_bridged() -> Graph:
    # a-b-c triangle and d-e-f triangle joined by the single edge c-d
    g = Graph()
    for n in "abcdef":
        g.add_node(n)
    for u, v in [("a", "b"), ("b", "c"), ("c", "a"),
                 ("d", "e"), ("e", "f"), ("f", "d"), ("c", "d")]:
        g.add_edge(u, v)
    return g


class TestPointsAndBridges:
    def test_the_joining_edge_is_the_only_bridge(self):
        art = Articulation(_two_triangles_bridged())
        assert art.bridges == {frozenset(("c", "d"))}

    def test_both_ends_of_the_bridge_are_articulation_points(self):
        art = Articulation(_two_triangles_bridged())
        assert art.points == {"c", "d"}

    def test_a_cycle_has_neither(self):
        g = Graph()
        for n in "abcd":
            g.add_node(n)
        for u, v in [("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")]:
            g.add_edge(u, v)
        art = Articulation(g)
        assert art.is_two_connected()

    def test_every_edge_of_a_path_is_a_bridge(self):
        g = Graph()
        for n in "abc":
            g.add_node(n)
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        art = Articulation(g)
        assert len(art.bridges) == 2
        assert art.points == {"b"}

    def test_a_root_with_two_children_is_an_articulation_point(self):
        # the first node visited is the hub of a star: it must still be found
        g = Graph()
        g.add_node("hub")
        for leaf in "ab":
            g.add_node(leaf)
            g.add_edge("hub", leaf)
        assert "hub" in Articulation(g).points


class TestRefusal:
    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            Articulation(Graph(directed=True))


class TestAgainstRemoval:
    def test_points_and_bridges_match_removing_each_and_recounting(self):
        rng = random.Random(23)
        for _ in range(30):
            g = Graph()
            nodes = [str(i) for i in range(8)]
            for n in nodes:
                g.add_node(n)
            for u in nodes:
                for v in nodes:
                    if u < v and rng.random() < 0.3:
                        g.add_edge(u, v)
            base = ConnectedComponents(g).count()
            art = Articulation(g)
            for n in nodes:
                splits = ConnectedComponents(_without_node(g, n)).count() > base
                assert (n in art.points) == splits
            for u, v, _w in g.edges():
                splits = ConnectedComponents(_without_edge(g, u, v)).count() > base
                assert (frozenset((u, v)) in art.bridges) == splits


class TestReport:
    def test_the_note_counts_both(self):
        note = Articulation(_two_triangles_bridged()).note()
        assert "2 articulation point(s), 1 bridge(s)" in note
