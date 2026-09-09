from __future__ import annotations

import random
from itertools import combinations

import pytest

from mesh.commutetime import CommuteTime
from mesh.errors import Invalid, Unreachable
from mesh.factories import complete, cycle, path, star
from mesh.graph import Graph


def _random_connected(seed: int, n: int, p: float) -> Graph:
    rng = random.Random(seed)
    g = path(n)
    for a, b in combinations(g.nodes(), 2):
        if not g.has_edge(a, b) and rng.random() < p:
            g.add_edge(a, b)
    return g


class TestClosedForms:
    def test_end_to_end_on_a_path_is_n_minus_one_squared(self):
        for n in (2, 3, 5, 8):
            ct = CommuteTime(path(n))
            assert ct.hitting("0", str(n - 1)) == pytest.approx((n - 1) ** 2)

    def test_any_pair_of_a_complete_graph_hits_in_n_minus_one(self):
        ct = CommuteTime(complete(6))
        assert ct.hitting("0", "3") == pytest.approx(5.0)
        assert ct.commute("0", "3") == pytest.approx(10.0)

    def test_neighbors_on_a_cycle_commute_in_two_n_minus_one(self):
        n = 7
        ct = CommuteTime(cycle(n))
        assert ct.commute("0", "1") == pytest.approx(2 * (n - 1))

    def test_a_star_is_asymmetric_leaf_to_hub_is_one_step(self):
        ct = CommuteTime(star(4))
        assert ct.hitting("1", "0") == pytest.approx(1.0)
        assert ct.hitting("0", "1") > 1.0
        # 2m times the unit resistance between hub and leaf: 8 * 1
        assert ct.commute("0", "1") == pytest.approx(8.0)


class TestResistanceRule:
    def test_commute_equals_twice_edges_times_resistance_on_random_graphs(self):
        for seed in range(811, 821):
            g = _random_connected(seed, 7, 0.3)
            ct = CommuteTime(g)
            for a, b in [("0", "6"), ("2", "4"), ("1", "5")]:
                assert ct.commute_matches_resistance(a, b)

    def test_resistance_of_a_path_is_the_hop_count_and_a_node_to_itself_is_zero(self):
        ct = CommuteTime(path(5))
        assert ct.resistance("0", "4") == pytest.approx(4.0)
        assert ct.resistance("2", "2") == 0.0

    def test_parallel_routes_lower_the_resistance(self):
        assert CommuteTime(cycle(4)).resistance("0", "2") == pytest.approx(1.0)


class TestRefusal:
    def test_unreachable_targets_and_bad_names_are_refused(self):
        g = Graph()
        for n in "abc":
            g.add_node(n)
        g.add_edge("a", "b")
        ct = CommuteTime(g)
        with pytest.raises(Unreachable):
            ct.hitting("a", "c")
        with pytest.raises(Invalid):
            ct.hitting("a", "zz")
        with pytest.raises(Invalid):
            CommuteTime(Graph(directed=True))


class TestReport:
    def test_the_note_shows_both_directions_and_the_rule(self):
        note = CommuteTime(star(3)).note("1", "0")
        assert note.startswith("hit 1->0 in 1.000 steps")
        assert "commute 6.000 against 2m R = 6.000" in note
