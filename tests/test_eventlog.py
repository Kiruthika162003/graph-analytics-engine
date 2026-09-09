from __future__ import annotations

import pytest

from mesh.errors import Invalid
from mesh.eventlog import EventLog
from mesh.graphcache import fingerprint


def _session() -> EventLog:
    log = EventLog()
    for n in ("ada", "ben", "cal"):
        log.add_node(n)
    log.add_edge("ada", "ben", 2.0)
    log.add_edge("ben", "cal")
    log.remove_edge("ben", "ada")
    log.add_node("dee")
    log.add_edge("cal", "dee")
    log.remove_node("ada")
    return log


class TestReplay:
    def test_replaying_everything_gives_the_live_graph(self):
        log = _session()
        assert fingerprint(log.replay()) == fingerprint(log.graph)
        assert sorted(log.graph.nodes()) == ["ben", "cal", "dee"]
        assert log.graph.edge_count() == 2

    def test_replaying_to_an_earlier_point_gives_the_graph_as_it_was(self):
        log = _session()
        early = log.replay(upto=5)
        assert sorted(early.nodes()) == ["ada", "ben", "cal"]
        assert early.has_edge("ada", "ben")
        assert early.weight("ada", "ben") == 2.0
        assert log.replay(upto=0).node_count() == 0

    def test_sequence_numbers_are_contiguous_from_one(self):
        # the guess was eight events; three nodes, two edges, a removal, a node, an
        # edge, and a node removal make nine
        log = _session()
        assert log.contiguous()
        assert log.events[0][0] == 1
        assert log.events[-1][0] == 9


class TestRefusals:
    def test_bad_additions_and_removals_are_refused_by_name(self):
        log = EventLog()
        log.add_node("a")
        with pytest.raises(Invalid, match="already present"):
            log.add_node("a")
        with pytest.raises(Invalid, match="names 'b', which is absent"):
            log.add_edge("a", "b")
        with pytest.raises(Invalid, match="no edge a-a"):
            log.remove_edge("a", "a")
        with pytest.raises(Invalid, match="no node 'zz'"):
            log.remove_node("zz")
        with pytest.raises(Invalid):
            log.replay(upto=9)


class TestText:
    def test_a_round_trip_through_text_gives_an_equal_log(self):
        log = _session()
        text = log.dump()
        back = EventLog.load(text)
        assert back.events == log.events
        assert fingerprint(back.graph) == fingerprint(log.graph)
        assert text.startswith("undirected\n1 add_node ada\n")

    def test_a_directed_log_survives_the_round_trip(self):
        log = EventLog(directed=True)
        log.add_node("p")
        log.add_node("q")
        log.add_edge("p", "q", 3.5)
        back = EventLog.load(log.dump())
        assert back.directed
        assert back.graph.weight("p", "q") == 3.5

    def test_corrupt_text_is_refused_with_its_line(self):
        with pytest.raises(Invalid, match="line 1"):
            EventLog.load("sideways\n")
        with pytest.raises(Invalid, match="line 2 is not an event"):
            EventLog.load("undirected\nnonsense\n")
        with pytest.raises(Invalid, match="line 3 is malformed"):
            EventLog.load("undirected\n1 add_node a\n2 add_edge a\n")
        with pytest.raises(Invalid, match="unknown event 'teleport'"):
            EventLog.load("undirected\n1 teleport a b\n")


class TestReport:
    def test_the_note_counts_events_and_the_live_graph(self):
        note = _session().note()
        assert note.startswith("9 event(s); live graph 3 node(s) and 2 edge(s), fingerprint ")
