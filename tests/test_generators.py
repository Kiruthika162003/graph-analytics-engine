from __future__ import annotations

import pytest

from mesh.connectedcomponents import ConnectedComponents
from mesh.errors import Invalid
from mesh.generators import (
    barabasi_albert,
    erdos_renyi,
    fingerprint,
    note,
    watts_strogatz,
)


class TestErdosRenyi:
    def test_p_zero_has_no_edges_and_p_one_is_complete(self):
        assert erdos_renyi(6, 0.0).edge_count() == 0
        assert erdos_renyi(6, 1.0).edge_count() == 15

    def test_the_edge_count_tracks_p_times_the_pair_count(self):
        g = erdos_renyi(60, 0.3, seed=1)
        expected = 0.3 * 60 * 59 / 2
        assert abs(g.edge_count() - expected) < 0.25 * expected

    def test_the_same_seed_reproduces_the_graph(self):
        a = erdos_renyi(20, 0.4, seed=7).edges()
        b = erdos_renyi(20, 0.4, seed=7).edges()
        assert a == b

    def test_degrees_stay_close_to_the_mean(self):
        f = fingerprint(erdos_renyi(200, 0.1, seed=2))
        assert f["max_over_mean"] < 2.0


class TestBarabasiAlbert:
    def test_every_later_node_brings_exactly_m_edges(self):
        g = barabasi_albert(50, 3, seed=3)
        # core of 4 nodes has 6 edges, then 46 nodes bring 3 each
        assert g.edge_count() == 6 + 46 * 3

    def test_the_graph_is_connected(self):
        assert ConnectedComponents(barabasi_albert(80, 2, seed=4)).is_connected()

    def test_preferential_attachment_grows_hubs(self):
        ba = fingerprint(barabasi_albert(300, 2, seed=5))
        er = fingerprint(erdos_renyi(300, 4 / 299, seed=5))
        assert ba["max_over_mean"] > 2 * er["max_over_mean"]


class TestWattsStrogatz:
    def test_beta_zero_is_a_ring_with_full_clustering_pattern(self):
        g = watts_strogatz(20, 4, 0.0)
        assert g.edge_count() == 40
        # each node's k=4 neighbors on a ring share 3 of 6 possible edges
        assert fingerprint(g)["clustering"] == pytest.approx(0.5)

    def test_light_rewiring_keeps_clustering_far_above_random(self):
        ws = fingerprint(watts_strogatz(100, 6, 0.05, seed=6))
        er = fingerprint(erdos_renyi(100, 6 / 99, seed=6))
        assert ws["clustering"] > 3 * er["clustering"]

    def test_full_rewiring_destroys_the_clustering(self):
        ring = fingerprint(watts_strogatz(100, 6, 0.0))
        shuffled = fingerprint(watts_strogatz(100, 6, 1.0, seed=8))
        assert shuffled["clustering"] < ring["clustering"] / 2


class TestRefusals:
    def test_a_probability_outside_the_unit_interval_is_refused(self):
        with pytest.raises(Invalid):
            erdos_renyi(5, 1.5)

    def test_too_many_attachments_per_node_is_refused(self):
        with pytest.raises(Invalid):
            barabasi_albert(5, 5)

    def test_an_odd_ring_degree_is_refused(self):
        with pytest.raises(Invalid):
            watts_strogatz(10, 3, 0.1)


class TestReport:
    def test_the_note_carries_the_label_and_fingerprint(self):
        text = note(erdos_renyi(30, 0.2, seed=9), "erdos-renyi")
        assert text.startswith("erdos-renyi:")
        assert "clustering" in text
