from __future__ import annotations

import pytest

from mesh.errors import Invalid
from mesh.factories import complete
from mesh.generators import erdos_renyi, watts_strogatz
from mesh.graph import Graph
from mesh.smallworld import SmallWorld


class TestRegimes:
    def test_a_lightly_rewired_ring_is_a_small_world(self):
        sw = SmallWorld(watts_strogatz(60, 6, 0.1, seed=3), seed=3)
        assert sw.sigma() > 1.5
        assert sw.verdict() == "small world"

    def test_a_random_graph_scores_sigma_near_one(self):
        sw = SmallWorld(erdos_renyi(60, 0.12, seed=5), seed=5)
        assert 0.4 < sw.sigma() < 2.0

    def test_a_pure_ring_lattice_is_clustered_but_far_apart(self):
        sw = SmallWorld(watts_strogatz(40, 4, 0.0), seed=7)
        assert sw.clustering > sw.random_clustering
        assert sw.path_length > sw.random_path_length
        assert "lattice-like" in sw.verdict() or sw.omega() < 0

    def test_the_random_twin_matches_node_and_edge_counts_roughly(self):
        g = watts_strogatz(50, 4, 0.2, seed=9)
        sw = SmallWorld(g, seed=9)
        assert sw.random_twin.node_count() == g.node_count()
        assert abs(sw.random_twin.edge_count() - g.edge_count()) < 0.3 * g.edge_count()

    def test_a_disconnected_input_is_measured_on_its_largest_component(self):
        g = watts_strogatz(30, 4, 0.05, seed=11)
        g.add_node("island")
        g.add_node("rock")
        g.add_edge("island", "rock")
        sw = SmallWorld(g, seed=11)
        assert sw.trimmed
        assert "on the largest component" in sw.note()


class TestRefusals:
    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            SmallWorld(Graph(directed=True))

    def test_a_tiny_graph_is_refused(self):
        with pytest.raises(Invalid):
            SmallWorld(complete(3))


class TestReport:
    def test_the_note_carries_both_coefficients_and_the_verdict(self):
        note = SmallWorld(watts_strogatz(40, 4, 0.1, seed=13), seed=13).note()
        assert "sigma" in note and "omega" in note
        assert "clustering" in note
