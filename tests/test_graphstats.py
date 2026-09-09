from __future__ import annotations

import pytest

from mesh.errors import Invalid
from mesh.factories import complete, cycle, star
from mesh.graph import Graph
from mesh.graphstats import DegreeStats, power_law_graph


class TestMoments:
    def test_a_regular_graph_has_no_variance_and_branching_of_degree_minus_one(self):
        ds = DegreeStats(cycle(8))
        assert ds.mean() == 2.0
        assert ds.variance() == 0.0
        assert ds.branching() == 1.0
        assert not ds.heavy_tailed()

    def test_a_star_is_heavy_tailed_with_one_hub(self):
        ds = DegreeStats(star(30))
        assert ds.degrees[0] == 30
        assert ds.heavy_tailed()
        assert ds.ccdf()[-1] == (30, pytest.approx(1 / 31))

    def test_the_ccdf_starts_at_one_and_falls(self):
        ds = DegreeStats(star(4))
        curve = ds.ccdf()
        assert curve[0][1] == 1.0
        assert [f for _k, f in curve] == sorted((f for _k, f in curve), reverse=True)

    def test_an_empty_graph_reads_zero(self):
        ds = DegreeStats(Graph())
        assert ds.mean() == 0.0
        assert ds.branching() == 0.0
        assert ds.ccdf() == []


class TestPowerLaw:
    def test_the_fit_recovers_the_exponent_a_graph_was_built_from(self):
        g = power_law_graph(3000, gamma=2.5, kmin=3, seed=1)
        ds = DegreeStats(g)
        gamma = ds.power_law_exponent(3)
        assert 2.2 < gamma < 2.9
        assert ds.heavy_tailed()

    def test_the_fit_refuses_a_thin_tail(self):
        with pytest.raises(Invalid):
            DegreeStats(complete(4)).power_law_exponent(10)
        with pytest.raises(Invalid):
            DegreeStats(complete(4)).power_law_exponent(0)

    def test_the_builder_refuses_bad_parameters(self):
        with pytest.raises(Invalid):
            power_law_graph(10, gamma=1.0, kmin=2, seed=0)
        with pytest.raises(Invalid):
            power_law_graph(1, gamma=2.5, kmin=2, seed=0)


class TestReport:
    def test_the_note_reads_the_tail_and_the_exponent_when_asked(self):
        ds = DegreeStats(cycle(6))
        assert ds.note() == (
            "mean degree 2.00, variance 0.00 against Poisson 2.00; a light tail, branching 1.00"
        )
        assert "no exponent" in ds.note(kmin=5)
        heavy = DegreeStats(power_law_graph(500, gamma=2.5, kmin=2, seed=2)).note(kmin=2)
        assert "exponent" in heavy and "tail node(s)" in heavy
