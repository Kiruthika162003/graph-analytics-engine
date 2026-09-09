from __future__ import annotations

import random
from itertools import combinations

import pytest

from mesh.epidemic import Epidemic
from mesh.errors import Invalid
from mesh.factories import complete, cycle, path, star
from mesh.graph import Graph


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


class TestExactEnds:
    def test_no_transmission_infects_only_the_seed(self):
        ep = Epidemic(cycle(8), beta=0.0, gamma=0.5, seed=1)
        ever, peak, _rounds = ep.run("0")
        assert ever == 1
        assert peak == 1
        assert ep.final_size(trials=5) == pytest.approx(1 / 8)

    def test_certain_transmission_and_recovery_sweep_a_path_one_hop_per_round(self):
        ep = Epidemic(path(6), beta=1.0, gamma=1.0, seed=2)
        ever, peak, rounds = ep.run("0")
        assert ever == 6
        assert peak == 1
        assert rounds == 6
        assert ep.final_size(trials=3, patient_zero="0") == pytest.approx(1.0)

    def test_certain_transmission_on_a_star_peaks_at_the_leaves(self):
        ep = Epidemic(star(5), beta=1.0, gamma=1.0, seed=3)
        ever, peak, _rounds = ep.run("0")
        assert ever == 6
        assert peak == 5

    def test_the_peak_never_exceeds_the_final_count(self):
        ep = Epidemic(_random_graph(1009, 20, 0.2), beta=0.4, gamma=0.3, seed=4)
        for _ in range(10):
            ever, peak, _rounds = ep.run()
            assert peak <= ever


class TestThreshold:
    def test_a_dense_graph_takes_off_at_a_lower_beta_than_a_cycle(self):
        dense = Epidemic(complete(12), 0.5, 0.5, seed=5).observed_threshold(steps=10, trials=15)
        ring = Epidemic(cycle(12), 0.5, 0.5, seed=5).observed_threshold(steps=10, trials=15)
        assert dense < ring

    def test_the_spectral_threshold_is_gamma_over_the_radius(self):
        ep = Epidemic(complete(5), 0.2, 0.4)
        assert ep.spectral_threshold() == pytest.approx(0.4 / 4)
        assert Epidemic(cycle(6), 0.2, 0.6).spectral_threshold() == pytest.approx(0.3)

    def test_a_directed_graph_follows_arcs_and_has_no_spectral_estimate(self):
        g = Graph(directed=True)
        for n in "abc":
            g.add_node(n)
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        ep = Epidemic(g, 1.0, 1.0, seed=6)
        assert ep.run("a")[0] == 3
        assert ep.run("c")[0] == 1
        assert ep.spectral_threshold() is None
        assert "absent on a directed graph" in ep.note(trials=2)


class TestRefusal:
    def test_bad_rates_seeds_and_counts_are_refused(self):
        with pytest.raises(Invalid):
            Epidemic(cycle(3), beta=1.5, gamma=0.5)
        with pytest.raises(Invalid):
            Epidemic(cycle(3), beta=0.5, gamma=0.0)
        ep = Epidemic(cycle(3), 0.5, 0.5)
        with pytest.raises(Invalid):
            ep.run("zz")
        with pytest.raises(Invalid):
            ep.final_size(trials=0)

    def test_an_empty_graph_has_nothing_to_infect(self):
        ep = Epidemic(Graph(), 0.5, 0.5)
        assert ep.run() == (0, 0, 0)
        assert ep.final_size() == 0.0


class TestReport:
    def test_the_note_states_the_rates_the_size_and_the_threshold(self):
        note = Epidemic(complete(4), 0.5, 0.5, seed=7).note(trials=4)
        assert note.startswith("beta 0.5 gamma 0.5: mean final size")
        assert "spectral threshold for beta 0.167" in note
