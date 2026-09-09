from __future__ import annotations

import random
from itertools import combinations

import pytest

from mesh.bandwidth import Bandwidth
from mesh.errors import Invalid
from mesh.factories import complete, cycle, grid, path, star
from mesh.graph import Graph


class TestKnownValues:
    def test_a_path_has_bandwidth_one(self):
        b = Bandwidth(path(6))
        assert b.upper()[0] == 1
        assert b.lower() == 1
        assert b.exact() == 1

    def test_a_complete_graph_has_bandwidth_n_minus_one(self):
        # the guess was a lower reading of 2 from half the degree; the diameter floor,
        # n minus one over a diameter of one, is 4 and meets the exact value
        b = Bandwidth(complete(5))
        assert b.upper()[0] == 4
        assert b.lower() == 4
        assert b.exact() == 4

    def test_a_cycle_has_bandwidth_two(self):
        b = Bandwidth(cycle(7))
        assert b.exact() == 2
        assert b.upper()[0] == 2

    def test_a_star_has_bandwidth_half_the_leaves_rounded_up(self):
        # the guess was that the level ordering reaches 3 as well; it starts at a leaf
        # and puts the hub second, so the last leaf spans four. the optimum has the hub
        # in the middle, which no breadth-first ordering from a leaf can produce
        b = Bandwidth(star(5))
        assert b.exact() == 3
        assert b.lower() == 3
        assert b.upper()[0] == 4

    def test_a_three_by_three_grid_has_bandwidth_three(self):
        b = Bandwidth(grid(3))
        assert b.exact() == 3
        assert b.upper()[0] >= 3


class TestBounds:
    def test_the_two_readings_bracket_the_exact_value_on_random_graphs(self):
        rng = random.Random(599)
        for _ in range(15):
            g = Graph()
            nodes = [str(i) for i in range(7)]
            for n in nodes:
                g.add_node(n)
            for a, b in combinations(nodes, 2):
                if rng.random() < 0.4:
                    g.add_edge(a, b)
            bw = Bandwidth(g)
            upper, order = bw.upper()
            assert sorted(order) == sorted(nodes)
            assert bw.lower() <= bw.exact() <= upper
            assert Bandwidth.span(g, order) == upper

    def test_an_empty_graph_reads_zero_everywhere(self):
        b = Bandwidth(Graph())
        assert b.upper() == (0, [])
        assert b.lower() == 0
        assert b.exact() == 0


class TestRefusal:
    def test_a_directed_graph_and_a_large_exact_are_refused(self):
        with pytest.raises(Invalid):
            Bandwidth(Graph(directed=True))
        with pytest.raises(Invalid):
            Bandwidth(path(10)).exact()


class TestReport:
    def test_the_note_states_both_readings(self):
        assert "at most 1 from the level ordering, at least 1; the bounds meet" in (
            Bandwidth(path(5)).note()
        )
        assert "at most 4 from the level ordering, at least 4; the bounds meet" in (
            Bandwidth(complete(5)).note()
        )
        assert "a gap of 1" in Bandwidth(star(5)).note()
