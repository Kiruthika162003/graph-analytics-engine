from __future__ import annotations

import pytest

from mesh.errors import Invalid
from mesh.factories import cycle, path
from mesh.graph import Graph
from mesh.graphio import same_graph
from mesh.graphsequence import GraphSequence


def _edges(pairs: list[tuple[str, str]], nodes: str = "abcd") -> Graph:
    g = Graph()
    for n in nodes:
        g.add_node(n)
    for u, v in pairs:
        g.add_edge(u, v)
    return g


class TestConstantAndChurning:
    def test_a_constant_sequence_persists_fully_with_no_churn(self):
        seq = GraphSequence([cycle(5), cycle(5), cycle(5)])
        assert seq.persistence() == [1.0, 1.0]
        assert seq.churn() == [0, 0]
        assert same_graph(seq.core(), cycle(5))
        assert same_graph(seq.union(), cycle(5))
        assert seq.trend() == "steady"

    def test_replacing_every_edge_each_step_gives_zero_persistence_and_an_empty_core(self):
        seq = GraphSequence(
            [_edges([("a", "b"), ("c", "d")]), _edges([("a", "c"), ("b", "d")])]
        )
        assert seq.persistence() == [0.0]
        assert seq.core().edge_count() == 0
        assert seq.union().edge_count() == 4

    def test_node_churn_counts_arrivals_and_departures(self):
        seq = GraphSequence([_edges([("a", "b")], "ab"), _edges([("a", "c")], "ac")])
        assert seq.churn() == [2]


class TestContainment:
    def test_the_core_sits_inside_every_snapshot_and_every_snapshot_inside_the_union(self):
        snaps = [
            _edges([("a", "b"), ("b", "c"), ("c", "d")]),
            _edges([("a", "b"), ("b", "c"), ("a", "d")]),
            _edges([("a", "b"), ("c", "d"), ("b", "d")]),
        ]
        seq = GraphSequence(snaps)
        assert seq.core_inside_every_snapshot()
        union = {(min(u, v), max(u, v)) for u, v, _w in seq.union().edges()}
        for g in snaps:
            assert {(min(u, v), max(u, v)) for u, v, _w in g.edges()} <= union
        assert seq.core().edge_count() == 1

    def test_trend_reads_settling_and_churning(self):
        settling = GraphSequence(
            [
                _edges([("a", "b"), ("c", "d")]),
                _edges([("a", "c"), ("b", "d")]),
                _edges([("a", "c"), ("b", "d")]),
            ]
        )
        assert settling.trend() == "settling"
        churning = GraphSequence(
            [
                _edges([("a", "b"), ("c", "d")]),
                _edges([("a", "b"), ("c", "d")]),
                _edges([("a", "c"), ("b", "d")]),
            ]
        )
        assert churning.trend() == "churning"


class TestReports:
    def test_each_step_yields_a_change_report(self):
        seq = GraphSequence([path(3), cycle(3)])
        reports = seq.reports()
        assert len(reports) == 1
        assert reports[0][0] == "nodes +0 -0, edges +1 -0"

    def test_a_single_snapshot_has_no_steps(self):
        seq = GraphSequence([path(4)])
        assert seq.persistence() == []
        assert seq.reports() == []
        assert seq.trend() == "too short to read"


class TestRefusal:
    def test_empty_and_mixed_sequences_are_refused(self):
        with pytest.raises(Invalid):
            GraphSequence([])
        with pytest.raises(Invalid, match="snapshot 1"):
            GraphSequence([Graph(), Graph(directed=True)])


class TestReport:
    def test_the_note_lists_the_series(self):
        note = GraphSequence([cycle(4), cycle(4)]).note()
        assert note.startswith("2 snapshot(s); persistence [1.00], churn [0], core of 4")
        assert note.endswith("union of 4; too short to read")
