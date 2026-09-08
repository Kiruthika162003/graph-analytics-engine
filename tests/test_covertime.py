from __future__ import annotations

import pytest

from mesh.covertime import CoverTime
from mesh.errors import Invalid, Missing
from mesh.factories import complete, path, star
from mesh.graph import Graph


def _lollipop(clique: int, tail: int) -> Graph:
    g = complete(clique)
    prev = "0"
    for i in range(tail):
        node = f"t{i}"
        g.add_node(node)
        g.add_edge(prev, node)
        prev = node
    return g


class TestEstimate:
    def test_a_complete_graph_sits_near_the_coupon_collector_floor(self):
        ct = CoverTime(complete(10), "0", walks=300, seed=1)
        # the exact expectation is (n-1) times the harmonic number of n-1
        harmonic = sum(1 / k for k in range(1, 10))
        assert abs(ct.estimate() - 9 * harmonic) < 4 * ct.standard_error() + 2

    def test_the_estimate_never_exceeds_the_commute_time_ceiling(self):
        for g, start in ((path(8), "0"), (star(6), "0"), (_lollipop(5, 4), "0")):
            ct = CoverTime(g, start, walks=100, seed=2)
            assert ct.within_bounds()

    def test_a_lollipop_takes_far_longer_than_a_complete_graph_of_the_same_size(self):
        quick = CoverTime(complete(9), "0", walks=100, seed=3).estimate()
        slow = CoverTime(_lollipop(5, 4), "0", walks=100, seed=3).estimate()
        assert slow > 3 * quick

    def test_starting_in_the_middle_of_a_path_costs_more_than_an_end(self):
        # first guess: the middle is closer to everything so it covers faster.
        # Measured: from an end the walk needs only to hit the far end, about
        # (n-1)^2 = 64 steps; from the middle it must first reach one end,
        # about 16 steps, and then cross the whole path, 64 more, so the
        # middle is slower. The assertion was flipped by the measurement.
        end = CoverTime(path(9), "0", walks=200, seed=4).estimate()
        middle = CoverTime(path(9), "4", walks=200, seed=4).estimate()
        assert middle > end

    def test_the_same_seed_reproduces_the_samples(self):
        a = CoverTime(path(6), "0", walks=20, seed=9).samples
        b = CoverTime(path(6), "0", walks=20, seed=9).samples
        assert a == b

    def test_position_lies_between_floor_and_ceiling(self):
        ct = CoverTime(star(5), "0", walks=100, seed=5)
        assert 0.0 <= ct.position() <= 1.0


class TestRefusals:
    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            CoverTime(Graph(directed=True), "a")

    def test_a_missing_start_is_refused(self):
        with pytest.raises(Missing):
            CoverTime(path(3), "ghost")

    def test_a_disconnected_graph_is_refused(self):
        g = path(3)
        g.add_node("island")
        with pytest.raises(Invalid):
            CoverTime(g, "0")

    def test_zero_walks_is_refused(self):
        with pytest.raises(Invalid):
            CoverTime(path(3), "0", walks=0)


class TestReport:
    def test_the_note_states_the_estimate_and_bounds(self):
        note = CoverTime(path(5), "0", walks=50, seed=6).note()
        assert "cover time about" in note
        assert "ceiling of" in note
