from __future__ import annotations

import random

import pytest

from mesh.errors import Cyclic, Invalid
from mesh.graph import Graph
from mesh.transitiveclosure import TransitiveClosure


def _chain_with_shortcut() -> Graph:
    # a -> b -> c, plus the implied shortcut a -> c
    g = Graph(directed=True)
    for n in "abc":
        g.add_node(n)
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    g.add_edge("a", "c")
    return g


class TestClosure:
    def test_a_reaches_everything_downstream(self):
        tc = TransitiveClosure(_chain_with_shortcut())
        assert tc.reach["a"] == {"b", "c"}

    def test_the_end_of_the_chain_reaches_nothing(self):
        tc = TransitiveClosure(_chain_with_shortcut())
        assert tc.reach["c"] == set()

    def test_reaches_is_a_lookup_into_the_closure(self):
        tc = TransitiveClosure(_chain_with_shortcut())
        assert tc.reaches("a", "c")
        assert not tc.reaches("c", "a")

    def test_a_cycle_reaches_itself(self):
        g = Graph(directed=True)
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b")
        g.add_edge("b", "a")
        assert TransitiveClosure(g).reaches("a", "a")


class TestReduction:
    def test_the_implied_shortcut_is_removed(self):
        tc = TransitiveClosure(_chain_with_shortcut())
        assert sorted(tc.reduction()) == [("a", "b"), ("b", "c")]
        assert tc.removed_count() == 1

    def test_the_reduction_preserves_reachability(self):
        g = _chain_with_shortcut()
        tc = TransitiveClosure(g)
        reduced = Graph(directed=True)
        for n in g.nodes():
            reduced.add_node(n)
        for u, v in tc.reduction():
            reduced.add_edge(u, v)
        assert TransitiveClosure(reduced).reach == tc.reach

    def test_a_cyclic_graph_has_no_unique_reduction(self):
        g = Graph(directed=True)
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b")
        g.add_edge("b", "a")
        with pytest.raises(Cyclic):
            TransitiveClosure(g).reduction()


class TestRefusal:
    def test_an_undirected_graph_is_refused(self):
        with pytest.raises(Invalid):
            TransitiveClosure(Graph())


class TestAgainstBruteForce:
    def test_reduction_is_minimal_and_reachability_preserving_on_random_dags(self):
        rng = random.Random(97)
        for _ in range(25):
            g = Graph(directed=True)
            nodes = [str(i) for i in range(7)]
            for n in nodes:
                g.add_node(n)
            for i, u in enumerate(nodes):
                for v in nodes[i + 1 :]:
                    if rng.random() < 0.45:
                        g.add_edge(u, v)
            tc = TransitiveClosure(g)
            kept = tc.reduction()
            reduced = Graph(directed=True)
            for n in nodes:
                reduced.add_node(n)
            for u, v in kept:
                reduced.add_edge(u, v)
            # same closure, and removing any kept edge would break it
            assert TransitiveClosure(reduced).reach == tc.reach
            for u, v in kept:
                smaller = Graph(directed=True)
                for n in nodes:
                    smaller.add_node(n)
                for a, b in kept:
                    if (a, b) != (u, v):
                        smaller.add_edge(a, b)
                assert TransitiveClosure(smaller).reach != tc.reach


class TestReport:
    def test_the_note_states_the_removed_fraction(self):
        assert "removed 1 of 3" in TransitiveClosure(_chain_with_shortcut()).note()
