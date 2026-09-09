from __future__ import annotations

import random
from itertools import combinations
from math import cos, pi

import pytest

from mesh.errors import Invalid
from mesh.factories import complete, cycle, path, star
from mesh.graph import Graph
from mesh.graphenergy import GraphEnergy


def _random_graph(seed: int, n: int, p: float) -> Graph:
    rng = random.Random(seed)
    g = Graph()
    nodes = [str(i) for i in range(n)]
    for node in nodes:
        g.add_node(node)
    for a, b in combinations(nodes, 2):
        if rng.random() < p:
            g.add_edge(a, b)
    return g


class TestEnergy:
    def test_a_complete_graph_has_energy_two_n_minus_two(self):
        for n in (3, 4, 6):
            assert GraphEnergy(complete(n)).energy() == pytest.approx(2 * (n - 1))

    def test_an_edgeless_graph_has_no_energy_and_estrada_equal_to_n(self):
        g = Graph()
        for n in "abc":
            g.add_node(n)
        ge = GraphEnergy(g)
        assert ge.energy() == 0.0
        assert ge.estrada() == pytest.approx(3.0)

    def test_a_star_has_energy_twice_root_leaves(self):
        assert GraphEnergy(star(9)).energy() == pytest.approx(6.0)

    def test_a_cycle_energy_matches_the_cosine_sum(self):
        n = 8
        expected = sum(abs(2 * cos(2 * pi * k / n)) for k in range(n))
        assert GraphEnergy(cycle(n)).energy() == pytest.approx(expected)


class TestSpectralRadius:
    def test_the_radius_sits_between_average_and_maximum_degree(self):
        for seed in range(683, 693):
            assert GraphEnergy(_random_graph(seed, 8, 0.4)).radius_bounds_hold()

    def test_a_regular_graph_has_radius_equal_to_its_degree(self):
        assert GraphEnergy(cycle(7)).spectral_radius() == pytest.approx(2.0)
        assert GraphEnergy(complete(5)).spectral_radius() == pytest.approx(4.0)


class TestWalkCounts:
    def test_squares_count_edges_and_cubes_count_triangles_on_random_graphs(self):
        for seed in range(697, 707):
            ge = GraphEnergy(_random_graph(seed, 8, 0.45))
            assert ge.walk_identities_hold()

    def test_a_triangle_free_graph_has_zero_cubes(self):
        assert GraphEnergy(cycle(6)).triangles_from_spectrum() == pytest.approx(0.0, abs=1e-9)
        assert GraphEnergy(path(5)).closed_walks(3) == pytest.approx(0.0, abs=1e-9)

    def test_k4_has_four_triangles_from_the_spectrum(self):
        assert GraphEnergy(complete(4)).triangles_from_spectrum() == pytest.approx(4.0)

    def test_a_negative_walk_length_is_refused(self):
        with pytest.raises(Invalid):
            GraphEnergy(cycle(3)).closed_walks(-1)


class TestRefusal:
    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            GraphEnergy(Graph(directed=True))

    def test_an_empty_graph_reads_zero(self):
        ge = GraphEnergy(Graph())
        assert ge.energy() == 0.0
        assert ge.spectral_radius() == 0.0
        assert ge.radius_bounds_hold()


class TestReport:
    def test_the_note_states_the_readings_and_the_counts(self):
        note = GraphEnergy(complete(4)).note()
        assert "energy 6.0000" in note
        assert "spectral radius 3.0000" in note
        assert "counts 12 closed two-walks and 4 triangle(s)" in note
