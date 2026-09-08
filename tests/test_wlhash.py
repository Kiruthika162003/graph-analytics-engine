from __future__ import annotations

import random
from itertools import combinations

import pytest

from mesh.errors import Invalid
from mesh.graph import Graph
from mesh.wlhash import WLHash


def _relabel(g: Graph, names: dict[str, str], order: list[str]) -> Graph:
    h = Graph(directed=g.directed)
    for n in order:
        h.add_node(names[n])
    for u, v, w in g.edges():
        h.add_edge(names[u], names[v], w)
    return h


def _cycle(k: int, prefix: str) -> Graph:
    g = Graph()
    nodes = [f"{prefix}{i}" for i in range(k)]
    for n in nodes:
        g.add_node(n)
    for i in range(k):
        g.add_edge(nodes[i], nodes[(i + 1) % k])
    return g


class TestInvariance:
    def test_relabelling_and_reordering_leave_the_hash_unchanged(self):
        rng = random.Random(5)
        g = Graph()
        nodes = [str(i) for i in range(8)]
        for n in nodes:
            g.add_node(n)
        for a, b in combinations(nodes, 2):
            if rng.random() < 0.4:
                g.add_edge(a, b)
        names = {n: f"x{(int(n) * 5) % 8}" for n in nodes}
        order = nodes[:]
        rng.shuffle(order)
        assert WLHash(g).digest == WLHash(_relabel(g, names, order)).digest

    def test_a_different_shape_hashes_differently(self):
        path = Graph()
        for n in "abcd":
            path.add_node(n)
        for u, v in [("a", "b"), ("b", "c"), ("c", "d")]:
            path.add_edge(u, v)
        star = Graph()
        star.add_node("h")
        for leaf in "xyz":
            star.add_node(leaf)
            star.add_edge("h", leaf)
        assert WLHash(path).digest != WLHash(star).digest

    def test_different_sizes_never_collide(self):
        assert WLHash(_cycle(4, "a")).digest != WLHash(_cycle(5, "b")).digest


class TestLimit:
    def test_regular_graphs_of_one_degree_and_size_collide_as_documented(self):
        # two triangles and a hexagon are both 2-regular on six nodes
        two = Graph()
        for n in "abcdef":
            two.add_node(n)
        for u, v in combinations("abc", 2):
            two.add_edge(u, v)
        for u, v in combinations("def", 2):
            two.add_edge(u, v)
        hexagon = _cycle(6, "h")
        assert WLHash(two).digest == WLHash(hexagon).digest
        assert WLHash(two).distinct_colors() == 1

    def test_refinement_separates_nodes_by_role(self):
        star = Graph()
        star.add_node("h")
        for leaf in "xyz":
            star.add_node(leaf)
            star.add_edge("h", leaf)
        wl = WLHash(star)
        assert wl.distinct_colors() == 2
        assert wl.color["x"] == wl.color["y"] != wl.color["h"]

    def test_zero_rounds_is_the_degree_histogram(self):
        wl = WLHash(_cycle(5, "c"), rounds=0)
        assert wl.distinct_colors() == 1


class TestRefusal:
    def test_negative_rounds_are_refused(self):
        with pytest.raises(Invalid):
            WLHash(_cycle(3, "a"), rounds=-1)


class TestReport:
    def test_the_note_carries_the_digest_and_color_count(self):
        note = WLHash(_cycle(4, "a")).note()
        assert "digest" in note
        assert "1 distinct color(s)" in note
