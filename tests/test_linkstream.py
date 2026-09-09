from __future__ import annotations

import pytest

from mesh.errors import Invalid
from mesh.linkstream import LinkStream


def _office() -> LinkStream:
    return LinkStream(
        [
            ("ada", "ben", 1, 3),
            ("ben", "cal", 5, 7),
            ("cal", "dee", 2, 4),
            ("ada", "eve", 8, 9),
        ]
    )


class TestSnapshots:
    def test_a_snapshot_holds_the_links_alive_at_that_time(self):
        ls = _office()
        g = ls.snapshot(2.5)
        assert g.has_edge("ada", "ben")
        assert g.has_edge("cal", "dee")
        assert not g.has_edge("ben", "cal")
        assert ls.snapshot(20).edge_count() == 0

    def test_the_aggregate_weights_pairs_by_time_alive(self):
        g = _office().aggregate()
        assert g.edge_count() == 4
        assert g.weight("ada", "ben") == 2
        assert g.weight("ada", "eve") == 1

    def test_a_window_clips_the_weights(self):
        g = _office().aggregate(2, 6)
        assert g.weight("ada", "ben") == 1
        assert g.weight("ben", "cal") == 1
        assert not g.has_edge("ada", "eve")


class TestReach:
    def test_time_respecting_reach_waits_for_links_and_never_goes_back(self):
        ls = _office()
        reached = ls.reach("ada", 0)
        assert reached["ben"] == 1
        assert reached["cal"] == 5
        assert "dee" not in reached
        assert reached["eve"] == 8

    def test_the_static_aggregate_overstates_reach(self):
        ls = _office()
        assert "dee" in ls.aggregate_reach("ada")
        assert "dee" not in ls.reach("ada", 0)
        assert set(ls.reach("ada", 0)) <= ls.aggregate_reach("ada")

    def test_a_link_alive_only_before_the_start_never_helps(self):
        ls = _office()
        assert ls.reach("ada", 4) == {"ada": 4, "eve": 8}

    def test_starting_mid_link_uses_it_at_once(self):
        assert _office().reach("cal", 3)["dee"] == 3


class TestRefusal:
    def test_bad_links_and_unknown_sources_are_refused(self):
        with pytest.raises(Invalid):
            LinkStream([("a", "b", 5, 2)])
        with pytest.raises(Invalid):
            _office().reach("zz", 0)

    def test_an_instant_link_counts_in_a_window_that_holds_its_moment(self):
        ls = LinkStream([("a", "b", 3, 3)])
        assert ls.aggregate(2, 4).has_edge("a", "b")
        assert ls.snapshot(3).has_edge("a", "b")


class TestReport:
    def test_the_note_compares_the_two_reaches(self):
        note = _office().note("ada", 0)
        assert "4 node(s) reachable in time, 5 in the static aggregate" in note
        assert "4 link(s) overall" in note
