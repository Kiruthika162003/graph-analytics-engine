from __future__ import annotations

import random
from itertools import combinations, pairwise

import pytest

from mesh.errors import Invalid, Missing, Unreachable
from mesh.graph import Graph
from mesh.widestpath import WidestPath


def _links() -> Graph:
    # s-a-t is short but narrow (3), s-b-c-t is long but wide (8)
    g = Graph()
    for n in "sabct":
        g.add_node(n)
    g.add_edge("s", "a", 10)
    g.add_edge("a", "t", 3)
    g.add_edge("s", "b", 8)
    g.add_edge("b", "c", 9)
    g.add_edge("c", "t", 8)
    return g


def _brute_widest(g: Graph, s: str, t: str) -> float:
    best = float("-inf")

    def walk(node: str, seen: set[str], width: float) -> None:
        nonlocal best
        if node == t:
            best = max(best, width)
            return
        for nbr, cap in g.neighbors(node).items():
            if nbr not in seen:
                walk(nbr, seen | {nbr}, min(width, cap))

    walk(s, {s}, float("inf"))
    return best


class TestWidth:
    def test_the_wide_detour_beats_the_narrow_shortcut(self):
        wp = WidestPath(_links(), "s")
        assert wp.width_to("t") == 8
        assert wp.path_to("t") == ["s", "b", "c", "t"]

    def test_the_source_has_infinite_width_to_itself(self):
        assert WidestPath(_links(), "s").width_to("s") == float("inf")

    def test_the_path_width_is_its_narrowest_edge(self):
        g = _links()
        wp = WidestPath(g, "s")
        path = wp.path_to("t")
        assert min(g.weight(u, v) for u, v in pairwise(path)) == wp.width_to("t")

    def test_the_maximum_spanning_tree_gives_the_same_width(self):
        wp = WidestPath(_links(), "s")
        assert wp.maximum_spanning_tree_width("t") == wp.width_to("t")

    def test_direction_is_respected(self):
        g = Graph(directed=True)
        for n in "abc":
            g.add_node(n)
        g.add_edge("a", "b", 5)
        g.add_edge("b", "c", 2)
        assert WidestPath(g, "a").width_to("c") == 2
        with pytest.raises(Unreachable):
            WidestPath(g, "c").width_to("a")


class TestRefusals:
    def test_a_negative_capacity_is_refused(self):
        g = Graph()
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b", -1)
        with pytest.raises(Invalid):
            WidestPath(g, "a")

    def test_a_missing_source_is_refused(self):
        with pytest.raises(Missing):
            WidestPath(_links(), "ghost")

    def test_an_unreachable_node_is_refused(self):
        g = _links()
        g.add_node("island")
        with pytest.raises(Unreachable):
            WidestPath(g, "s").width_to("island")


class TestAgainstBruteForce:
    def test_widths_match_enumerating_every_simple_path(self):
        rng = random.Random(353)
        for _ in range(30):
            g = Graph()
            nodes = [str(i) for i in range(7)]
            for n in nodes:
                g.add_node(n)
            for a, b in combinations(nodes, 2):
                if rng.random() < 0.45:
                    g.add_edge(a, b, rng.randint(1, 20))
            wp = WidestPath(g, "0")
            for t in nodes[1:]:
                expected = _brute_widest(g, "0", t)
                if expected == float("-inf"):
                    with pytest.raises(Unreachable):
                        wp.width_to(t)
                else:
                    assert wp.width_to(t) == expected
                    assert wp.maximum_spanning_tree_width(t) == expected


class TestReport:
    def test_the_note_locates_the_bottleneck(self):
        note = WidestPath(_links(), "s").note("t")
        assert "carries 8" in note
        assert "somewhere in the middle" in note
