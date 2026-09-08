from __future__ import annotations

import random
from itertools import combinations

import pytest

from mesh.cyclecount import CycleCount
from mesh.errors import Invalid
from mesh.factories import complete, complete_bipartite, cycle, grid, path
from mesh.graph import Graph


class TestFormulas:
    def test_k4_has_four_triangles_and_three_four_cycles(self):
        cc = CycleCount(complete(4))
        assert cc.triangles == 4
        assert cc.four_cycles == 3

    def test_a_four_cycle_is_one_square_and_no_triangle(self):
        cc = CycleCount(cycle(4))
        assert (cc.triangles, cc.four_cycles) == (0, 1)

    def test_a_grid_has_one_square_per_cell(self):
        cc = CycleCount(grid(3))
        assert cc.triangles == 0
        assert cc.four_cycles == 4

    def test_a_path_has_no_cycles_at_all(self):
        cc = CycleCount(path(6))
        assert (cc.triangles, cc.four_cycles) == (0, 0)

    def test_complete_bipartite_counts_squares_by_pairs_of_pairs(self):
        # K3,3: choose 2 of 3 on each side, 3 * 3 = 9 squares
        cc = CycleCount(complete_bipartite(3, 3))
        assert cc.four_cycles == 9
        assert cc.triangles == 0


class TestAgainstEnumeration:
    def test_both_formulas_match_counting_on_random_graphs(self):
        rng = random.Random(397)
        for _ in range(30):
            g = Graph()
            nodes = [str(i) for i in range(8)]
            for n in nodes:
                g.add_node(n)
            for a, b in combinations(nodes, 2):
                if rng.random() < 0.45:
                    g.add_edge(a, b)
            cc = CycleCount(g)
            assert cc.triangles == cc.enumerate_triangles()
            assert cc.four_cycles == cc.enumerate_four_cycles()


class TestRefusals:
    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            CycleCount(Graph(directed=True))

    def test_enumeration_is_capped(self):
        with pytest.raises(Invalid):
            CycleCount(path(13)).enumerate_triangles()


class TestReport:
    def test_the_note_names_the_shape(self):
        assert "bipartite-like squares" in CycleCount(grid(3)).note()
        assert "tight triangles" in CycleCount(complete(4)).note()
