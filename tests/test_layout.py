from __future__ import annotations

from itertools import combinations, pairwise
from math import hypot

import pytest

from mesh.errors import Cyclic, Invalid
from mesh.factories import cycle, path
from mesh.graph import Graph
from mesh.layout import Layout


def _two_cliques() -> Graph:
    g = Graph()
    for group in ("abcd", "wxyz"):
        for n in group:
            g.add_node(n)
        for a, b in combinations(group, 2):
            g.add_edge(a, b)
    g.add_edge("d", "w")
    return g


def _dag() -> Graph:
    g = Graph(directed=True)
    for n in ("src", "a", "b", "c", "sink"):
        g.add_node(n)
    arcs = [("src", "a"), ("src", "b"), ("a", "c"), ("b", "c"), ("c", "sink"), ("b", "sink")]
    for u, v in arcs:
        g.add_edge(u, v)
    return g


class TestCircle:
    def test_every_node_sits_on_the_ring_at_equal_spacing(self):
        pos = Layout(cycle(6)).circle()
        assert len(pos) == 6
        radii = [hypot(x - 0.5, y - 0.5) for x, y in pos.values()]
        assert all(r == pytest.approx(0.45) for r in radii)
        ordered = [pos[str(i)] for i in range(6)]
        gaps = [hypot(a[0] - b[0], a[1] - b[1]) for a, b in pairwise(ordered)]
        assert all(g == pytest.approx(gaps[0]) for g in gaps)

    def test_an_empty_graph_has_no_positions(self):
        assert Layout(Graph()).circle() == {}


class TestSpring:
    def test_two_cliques_settle_into_two_clusters(self):
        g = _two_cliques()
        pos = Layout(g).spring(rounds=300, seed=1)
        assert Layout.inside_square(pos)

        def spread(names: str) -> float:
            pts = [pos[n] for n in names]
            return max(hypot(a[0] - b[0], a[1] - b[1]) for a, b in combinations(pts, 2))

        between = hypot(pos["a"][0] - pos["z"][0], pos["a"][1] - pos["z"][1])
        assert between > spread("abcd")
        assert between > spread("wxyz")

    def test_the_same_seed_reproduces_and_a_single_node_centers(self):
        g = path(4)
        assert Layout(g).spring(rounds=50, seed=3) == Layout(g).spring(rounds=50, seed=3)
        solo = Graph()
        solo.add_node("x")
        assert Layout(solo).spring() == {"x": (0.5, 0.5)}

    def test_negative_rounds_are_refused(self):
        with pytest.raises(Invalid):
            Layout(path(2)).spring(rounds=-1)


class TestLayered:
    def test_every_arc_points_strictly_downward(self):
        g = _dag()
        pos = Layout(g).layered()
        for u, v, _w in g.edges():
            assert pos[v][1] > pos[u][1]
        assert pos["src"][1] == 0.0
        assert pos["sink"][1] == 1.0

    def test_nodes_on_one_layer_are_spread_apart(self):
        pos = Layout(_dag()).layered()
        assert pos["a"][1] == pos["b"][1]
        assert pos["a"][0] != pos["b"][0]

    def test_cycles_and_undirected_graphs_are_refused(self):
        g = Graph(directed=True)
        for n in "ab":
            g.add_node(n)
        g.add_edge("a", "b")
        g.add_edge("b", "a")
        with pytest.raises(Cyclic):
            Layout(g).layered()
        with pytest.raises(Invalid):
            Layout(path(2)).layered()


class TestReport:
    def test_the_note_names_the_method_and_the_square_check(self):
        lay = Layout(cycle(3))
        note = lay.note(lay.circle(), "circle")
        assert note == "circle layout of 3 node(s), all inside the unit square: True"
