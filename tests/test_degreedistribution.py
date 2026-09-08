from __future__ import annotations

import math

import pytest

from mesh.degreedistribution import DegreeDistribution
from mesh.errors import Invalid
from mesh.generators import barabasi_albert, erdos_renyi
from mesh.graph import Graph


def _star(leaves: int) -> Graph:
    g = Graph()
    g.add_node("hub")
    for i in range(leaves):
        g.add_node(f"l{i}")
        g.add_edge("hub", f"l{i}")
    return g


class TestMoments:
    def test_the_histogram_counts_each_degree(self):
        dd = DegreeDistribution(_star(4))
        assert dd.histogram() == {1: 4, 4: 1}

    def test_mean_and_variance_follow_the_degrees(self):
        dd = DegreeDistribution(_star(4))
        # degrees 4,1,1,1,1: mean 1.6
        assert dd.mean() == pytest.approx(1.6)
        assert dd.variance() == pytest.approx(1.44)

    def test_a_regular_graph_has_zero_variance(self):
        g = Graph()
        for n in "abcd":
            g.add_node(n)
        for u, v in [("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")]:
            g.add_edge(u, v)
        assert DegreeDistribution(g).variance() == 0.0


class TestShape:
    def test_a_random_graph_has_a_variance_ratio_near_one(self):
        dd = DegreeDistribution(erdos_renyi(400, 0.02, seed=1))
        assert 0.6 < dd.variance_ratio() < 1.6

    def test_preferential_attachment_has_a_heavy_tail(self):
        ba = DegreeDistribution(barabasi_albert(400, 2, seed=2))
        er = DegreeDistribution(erdos_renyi(400, 4 / 399, seed=2))
        assert ba.variance_ratio() > 2 * er.variance_ratio()

    def test_the_power_law_exponent_lands_near_three_for_preferential_attachment(self):
        dd = DegreeDistribution(barabasi_albert(2000, 3, seed=3))
        exponent = dd.power_law_exponent(minimum=6)
        assert 2.2 < exponent < 3.8

    def test_the_estimator_matches_its_formula_on_a_small_tail(self):
        g = _star(3)
        dd = DegreeDistribution(g)
        # tail at minimum 1: degrees 3,1,1,1 -> 1 + 4 / sum(log(d / 0.5))
        expected = 1 + 4 / sum(math.log(d / 0.5) for d in [3, 1, 1, 1])
        assert dd.power_law_exponent(1) == pytest.approx(expected)


class TestRefusals:
    def test_an_empty_graph_is_refused(self):
        with pytest.raises(Invalid):
            DegreeDistribution(Graph())

    def test_a_tail_with_fewer_than_two_degrees_is_refused(self):
        with pytest.raises(Invalid):
            DegreeDistribution(_star(3)).power_law_exponent(minimum=3)


class TestReport:
    def test_the_note_names_the_shape(self):
        note = DegreeDistribution(_star(10)).note()
        assert "hub-heavy" in note
