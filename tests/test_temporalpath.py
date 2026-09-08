from __future__ import annotations

import pytest

from mesh.errors import Invalid, Missing, Unreachable
from mesh.temporalpath import TemporalGraph


def _flights() -> TemporalGraph:
    # a-b at 1, b-c at 2 (usable after a-b), c-d at 0 (too early to chain)
    t = TemporalGraph(directed=True)
    t.add_contact("a", "b", 1)
    t.add_contact("b", "c", 2)
    t.add_contact("c", "d", 0)
    return t


class TestEarliestArrival:
    def test_contacts_in_order_chain_into_a_route(self):
        arrival = _flights().earliest_arrival("a")
        assert arrival["b"] == 1
        assert arrival["c"] == 2

    def test_a_contact_that_happened_too_early_does_not_chain(self):
        arrival = _flights().earliest_arrival("a")
        assert "d" not in arrival

    def test_the_static_graph_promises_more_than_time_allows(self):
        t = _flights()
        assert t.static_reach("a") == 3
        assert t.temporal_reach("a") == 2

    def test_reachability_is_not_symmetric_even_on_undirected_contacts(self):
        t = TemporalGraph()
        t.add_contact("a", "b", 1)
        t.add_contact("b", "c", 2)
        assert "c" in t.earliest_arrival("a")
        assert "a" not in t.earliest_arrival("c")  # b-c at 2 then a-b at 1: no

    def test_a_later_start_time_forfeits_early_contacts(self):
        t = _flights()
        assert "b" in t.earliest_arrival("a", start=1)
        assert t.earliest_arrival("a", start=2) == {"a": 2}

    def test_the_route_is_a_timed_sequence(self):
        route = _flights().route("a", "c")
        assert route == [("a", 0), ("b", 1), ("c", 2)]

    def test_the_earliest_of_two_contacts_wins(self):
        t = TemporalGraph()
        t.add_contact("a", "b", 5)
        t.add_contact("a", "b", 3)
        assert t.earliest_arrival("a")["b"] == 3


class TestRefusals:
    def test_a_negative_time_is_refused(self):
        with pytest.raises(Invalid):
            TemporalGraph().add_contact("a", "b", -1)

    def test_an_unknown_source_is_refused(self):
        with pytest.raises(Missing):
            _flights().earliest_arrival("ghost")

    def test_an_unreached_target_is_refused(self):
        with pytest.raises(Unreachable):
            _flights().route("a", "d")


class TestReport:
    def test_the_note_contrasts_timed_and_static_reach(self):
        note = _flights().note("a")
        assert "2 node(s) reachable in time against 3" in note
