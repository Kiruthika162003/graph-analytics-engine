from __future__ import annotations

import random
from itertools import combinations

import pytest

from mesh.errors import Invalid
from mesh.feedbackarcset import FeedbackArcSet
from mesh.graph import Graph
from mesh.toposort import TopologicalSort


def _brute_min_feedback(g: Graph) -> int:
    edges = [(u, v) for u, v, _w in g.edges()]
    for size in range(len(edges) + 1):
        for gone in combinations(edges, size):
            h = Graph(directed=True)
            for n in g.nodes():
                h.add_node(n)
            gone_set = set(gone)
            for u, v in edges:
                if (u, v) not in gone_set:
                    h.add_edge(u, v)
            if TopologicalSort(h).is_acyclic():
                return size
    return len(edges)


class TestFeedback:
    def test_a_dag_needs_no_removal(self):
        g = Graph(directed=True)
        for n in "abc":
            g.add_node(n)
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        fas = FeedbackArcSet(g)
        assert fas.removed == []

    def test_a_single_cycle_loses_exactly_one_edge(self):
        g = Graph(directed=True)
        for n in "abc":
            g.add_node(n)
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        g.add_edge("c", "a")
        fas = FeedbackArcSet(g)
        assert len(fas.removed) == 1
        assert fas.remainder_is_acyclic()

    def test_the_remainder_is_always_acyclic(self):
        rng = random.Random(251)
        for _ in range(30):
            g = Graph(directed=True)
            nodes = [str(i) for i in range(8)]
            for n in nodes:
                g.add_node(n)
            for u in nodes:
                for v in nodes:
                    if u != v and rng.random() < 0.3:
                        g.add_edge(u, v)
            assert FeedbackArcSet(g).remainder_is_acyclic()

    def test_the_order_places_every_node_once(self):
        g = Graph(directed=True)
        for n in "abcd":
            g.add_node(n)
        g.add_edge("a", "b")
        g.add_edge("b", "a")
        g.add_edge("c", "d")
        fas = FeedbackArcSet(g)
        assert sorted(fas.order) == ["a", "b", "c", "d"]


class TestRefusal:
    def test_an_undirected_graph_is_refused(self):
        with pytest.raises(Invalid):
            FeedbackArcSet(Graph())


class TestAgainstBruteForce:
    def test_the_heuristic_is_measured_against_the_true_minimum(self):
        rng = random.Random(257)
        gaps = []
        for _ in range(25):
            g = Graph(directed=True)
            nodes = [str(i) for i in range(5)]
            for n in nodes:
                g.add_node(n)
            for u in nodes:
                for v in nodes:
                    if u != v and rng.random() < 0.4:
                        g.add_edge(u, v)
            found = len(FeedbackArcSet(g).removed)
            best = _brute_min_feedback(g)
            assert found >= best
            gaps.append(found - best)
        # the heuristic hits the optimum on most small instances
        assert gaps.count(0) >= len(gaps) // 2


class TestReport:
    def test_the_note_states_the_removed_share(self):
        g = Graph(directed=True)
        for n in "ab":
            g.add_node(n)
        g.add_edge("a", "b")
        g.add_edge("b", "a")
        assert "removed 1 of 2 edge(s) (50%)" in FeedbackArcSet(g).note()
