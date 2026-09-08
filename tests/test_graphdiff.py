from __future__ import annotations

import pytest

from mesh.errors import Invalid
from mesh.graph import Graph
from mesh.graphdiff import GraphDiff


def _before() -> Graph:
    g = Graph()
    for n in "abcd":
        g.add_node(n)
    for u, v in [("a", "b"), ("b", "c"), ("c", "d")]:
        g.add_edge(u, v)
    return g


def _after() -> Graph:
    # c-d is gone, a-c and d-e appear, e joins
    g = Graph()
    for n in "abcde":
        g.add_node(n)
    for u, v in [("a", "b"), ("b", "c"), ("a", "c"), ("d", "e")]:
        g.add_edge(u, v)
    return g


class TestEdges:
    def test_added_and_removed_edges_are_listed(self):
        d = GraphDiff(_before(), _after())
        assert d.added == [frozenset(("a", "c")), frozenset(("d", "e"))]
        assert d.removed == [frozenset(("c", "d"))]

    def test_undirected_edges_compare_regardless_of_orientation(self):
        a = Graph()
        b = Graph()
        for g in (a, b):
            g.add_node("x")
            g.add_node("y")
        a.add_edge("x", "y")
        b.add_edge("y", "x")
        d = GraphDiff(a, b)
        assert d.added == [] and d.removed == []
        assert d.similarity() == 1.0

    def test_directed_edges_compare_by_orientation(self):
        a = Graph(directed=True)
        b = Graph(directed=True)
        for g in (a, b):
            g.add_node("x")
            g.add_node("y")
        a.add_edge("x", "y")
        b.add_edge("y", "x")
        d = GraphDiff(a, b)
        assert d.added == [("y", "x")]
        assert d.removed == [("x", "y")]


class TestNodesAndDegrees:
    def test_joined_and_left_nodes_are_listed(self):
        d = GraphDiff(_before(), _after())
        assert d.joined == ["e"]
        assert d.left == []

    def test_degree_deltas_cover_the_common_nodes(self):
        d = GraphDiff(_before(), _after()).degree_delta()
        assert d == {"a": 1, "b": 0, "c": 0, "d": 0}

    def test_the_most_changed_node_ranks_first(self):
        assert GraphDiff(_before(), _after()).most_changed(1) == [("a", 1)]


class TestSummaries:
    def test_similarity_and_churn_are_complements(self):
        d = GraphDiff(_before(), _after())
        # shared a-b, b-c: 2 of a union of 5
        assert d.similarity() == pytest.approx(2 / 5)
        assert d.churn() == pytest.approx(3 / 5)
        assert d.similarity() + d.churn() == pytest.approx(1.0)

    def test_identical_graphs_have_similarity_one(self):
        d = GraphDiff(_before(), _before())
        assert d.similarity() == 1.0
        assert d.churn() == 0.0


class TestRefusal:
    def test_mixed_directedness_is_refused(self):
        with pytest.raises(Invalid):
            GraphDiff(Graph(), Graph(directed=True))


class TestReport:
    def test_the_note_states_counts_and_churn(self):
        note = GraphDiff(_before(), _after()).note()
        assert "2 edge(s) added, 1 removed" in note
        assert "1 node(s) with a changed degree" in note
