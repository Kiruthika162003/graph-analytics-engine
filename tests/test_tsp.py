from __future__ import annotations

import random
from itertools import permutations
from math import hypot

import pytest

from mesh.errors import Invalid
from mesh.factories import path
from mesh.graph import Graph
from mesh.tsp import TravellingSalesman


def _points(seed: int, n: int) -> Graph:
    rng = random.Random(seed)
    pts = {f"p{i}": (rng.uniform(0, 10), rng.uniform(0, 10)) for i in range(n)}
    g = Graph()
    for name in pts:
        g.add_node(name)
    names = list(pts)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            g.add_edge(a, b, hypot(pts[a][0] - pts[b][0], pts[a][1] - pts[b][1]))
    return g


def _brute(ts: TravellingSalesman) -> float:
    first = ts.nodes[0]
    return min(ts.tour_length([first, *p]) for p in permutations(ts.nodes[1:]))


class TestExact:
    def test_the_exact_tour_matches_brute_force_on_small_point_sets(self):
        for seed in range(919, 925):
            ts = TravellingSalesman(_points(seed, 6))
            tour = ts.exact()
            assert ts.is_tour(tour)
            assert ts.tour_length(tour) == pytest.approx(_brute(ts))

    def test_four_corners_of_a_square_tour_its_perimeter(self):
        g = Graph()
        corners = {"a": (0, 0), "b": (1, 0), "c": (1, 1), "d": (0, 1)}
        for n in corners:
            g.add_node(n)
        names = list(corners)
        for i, u in enumerate(names):
            for v in names[i + 1 :]:
                (xu, yu), (xv, yv) = corners[u], corners[v]
                g.add_edge(u, v, hypot(xu - xv, yu - yv))
        ts = TravellingSalesman(g)
        assert ts.tour_length(ts.exact()) == pytest.approx(4.0)

    def test_tiny_inputs(self):
        g = Graph()
        g.add_node("solo")
        ts = TravellingSalesman(g)
        assert ts.exact() == ["solo"]
        assert ts.tour_length(ts.exact()) == 0.0
        assert TravellingSalesman(Graph()).double_tree() == []


class TestDoubleTree:
    def test_the_shortcut_tour_visits_everyone_and_stays_within_twice_the_optimum(self):
        for seed in range(929, 941):
            ts = TravellingSalesman(_points(seed, 8))
            approx = ts.double_tree()
            assert ts.is_tour(approx)
            assert ts.ratio() <= 2.0 + 1e-9
            assert ts.ratio() >= 1.0 - 1e-9

    def test_points_on_a_line_give_a_ratio_of_exactly_one(self):
        g = Graph()
        for i in range(5):
            g.add_node(str(i))
        for i in range(5):
            for j in range(i + 1, 5):
                g.add_edge(str(i), str(j), float(j - i))
        ts = TravellingSalesman(g)
        assert ts.ratio() == pytest.approx(1.0)


class TestRefusal:
    def test_incomplete_directed_and_oversized_graphs_are_refused(self):
        with pytest.raises(Invalid, match="no edge 0-2"):
            TravellingSalesman(path(3))
        with pytest.raises(Invalid):
            TravellingSalesman(Graph(directed=True))
        with pytest.raises(Invalid):
            TravellingSalesman(_points(0, 13)).exact()


class TestReport:
    def test_the_note_gives_both_lengths_and_the_ratio(self):
        note = TravellingSalesman(_points(947, 5)).note()
        assert note.startswith("exact tour ")
        assert "double-tree tour" in note
        assert "ratio" in note
