from __future__ import annotations

import random
from itertools import combinations

import pytest

from mesh.errors import Invalid, Missing
from mesh.graph import Graph
from mesh.vertexseparator import VertexSeparator


def _hourglass() -> Graph:
    # two triangles joined at a single hub h: h separates a from d
    g = Graph()
    for n in "abhcd":
        g.add_node(n)
    for u, v in [("a", "b"), ("b", "h"), ("h", "a"), ("h", "c"), ("c", "d"), ("d", "h")]:
        g.add_edge(u, v)
    return g


class TestSeparator:
    def test_the_hub_of_an_hourglass_is_the_separator(self):
        vs = VertexSeparator(_hourglass(), "a", "d")
        assert vs.separator == {"h"}
        assert vs.separates()
        assert vs.is_minimum()

    def test_parallel_routes_need_every_middle_node(self):
        g = Graph()
        for n in "sxyzt":
            g.add_node(n)
        for mid in "xyz":
            g.add_edge("s", mid)
            g.add_edge(mid, "t")
        vs = VertexSeparator(g, "s", "t")
        assert vs.separator == {"x", "y", "z"}
        assert vs.is_minimum()

    def test_already_apart_needs_nothing(self):
        g = Graph()
        g.add_node("a")
        g.add_node("b")
        vs = VertexSeparator(g, "a", "b")
        assert vs.separator == set()
        assert "already apart" in vs.note()

    def test_direction_is_respected(self):
        g = Graph(directed=True)
        for n in "abc":
            g.add_node(n)
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        assert VertexSeparator(g, "a", "c").separator == {"b"}
        assert VertexSeparator(g, "c", "a").separator == set()


class TestRefusals:
    def test_adjacent_endpoints_are_refused(self):
        g = Graph()
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b")
        with pytest.raises(Invalid):
            VertexSeparator(g, "a", "b")

    def test_a_missing_endpoint_is_refused(self):
        with pytest.raises(Missing):
            VertexSeparator(_hourglass(), "a", "ghost")

    def test_equal_endpoints_are_refused(self):
        with pytest.raises(Invalid):
            VertexSeparator(_hourglass(), "a", "a")


class TestAgainstMenger:
    def test_the_separator_always_separates_and_is_always_minimum(self):
        rng = random.Random(409)
        checked = 0
        for _ in range(30):
            g = Graph()
            nodes = [str(i) for i in range(8)]
            for n in nodes:
                g.add_node(n)
            for a, b in combinations(nodes, 2):
                if rng.random() < 0.35:
                    g.add_edge(a, b)
            if g.has_edge("0", "7"):
                continue
            vs = VertexSeparator(g, "0", "7")
            assert vs.separates()
            assert vs.is_minimum()
            checked += 1
        assert checked > 10


class TestReport:
    def test_the_note_names_the_single_point_of_failure(self):
        assert "a single point of failure" in VertexSeparator(_hourglass(), "a", "d").note()
