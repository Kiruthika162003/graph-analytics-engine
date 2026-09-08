from __future__ import annotations

import random

import pytest

from mesh.bfs import BFS
from mesh.dominators import Dominators
from mesh.errors import Invalid, Missing
from mesh.graph import Graph


def _diamond_with_loop() -> Graph:
    # entry -> a -> b, entry -> a -> c, b -> d, c -> d, d -> a (loop), d -> exit
    g = Graph(directed=True)
    for n in ["entry", "a", "b", "c", "d", "exit"]:
        g.add_node(n)
    for u, v in [("entry", "a"), ("a", "b"), ("a", "c"), ("b", "d"),
                 ("c", "d"), ("d", "a"), ("d", "exit")]:
        g.add_edge(u, v)
    return g


def _without(g: Graph, gone: str) -> Graph:
    h = Graph(directed=True)
    for n in g.nodes():
        if n != gone:
            h.add_node(n)
    for u, v, w in g.edges():
        if gone not in (u, v):
            h.add_edge(u, v, w)
    return h


class TestImmediateDominators:
    def test_the_entry_dominates_itself(self):
        d = Dominators(_diamond_with_loop(), "entry")
        assert d.idom["entry"] == "entry"

    def test_the_merge_point_is_dominated_by_the_fork_not_a_branch(self):
        d = Dominators(_diamond_with_loop(), "entry")
        assert d.idom["d"] == "a"
        assert not d.dominates("b", "d")
        assert not d.dominates("c", "d")

    def test_dominators_form_a_chain_to_the_entry(self):
        d = Dominators(_diamond_with_loop(), "entry")
        assert d.dominators_of("exit") == ["exit", "d", "a", "entry"]

    def test_a_loop_back_edge_does_not_change_dominance(self):
        # d -> a exists, but a still dominates d, not the other way around
        d = Dominators(_diamond_with_loop(), "entry")
        assert d.dominates("a", "d")
        assert not d.dominates("d", "a")

    def test_a_straight_line_dominates_in_order(self):
        g = Graph(directed=True)
        for n in "abc":
            g.add_node(n)
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        d = Dominators(g, "a")
        assert d.dominators_of("c") == ["c", "b", "a"]
        assert d.tree_depth() == 2


class TestRefusals:
    def test_an_undirected_graph_is_refused(self):
        with pytest.raises(Invalid):
            Dominators(Graph(), "a")

    def test_a_missing_entry_is_refused(self):
        g = Graph(directed=True)
        g.add_node("a")
        with pytest.raises(Missing):
            Dominators(g, "ghost")

    def test_an_unreachable_node_has_no_dominators(self):
        g = _diamond_with_loop()
        g.add_node("orphan")
        with pytest.raises(Missing):
            Dominators(g, "entry").dominators_of("orphan")


class TestAgainstRemoval:
    def test_dominance_matches_removing_the_node_and_checking_reachability(self):
        rng = random.Random(151)
        for _ in range(30):
            g = Graph(directed=True)
            nodes = [str(i) for i in range(8)]
            for n in nodes:
                g.add_node(n)
            for u in nodes:
                for v in nodes:
                    # no edges into the entry, and no self loops
                    if v not in (u, "0") and rng.random() < 0.3:
                        g.add_edge(u, v)
            dom = Dominators(g, "0")
            reachable = BFS(g, "0").reachable_nodes()
            for d in reachable:
                if d == "0":
                    continue
                still = BFS(_without(g, d), "0").reachable_nodes()
                for n in reachable:
                    if n == d:
                        continue
                    # d dominates n exactly when removing d cuts n off
                    assert dom.dominates(d, n) == (n not in still)


class TestReport:
    def test_the_note_states_depth_and_passes(self):
        note = Dominators(_diamond_with_loop(), "entry").note()
        assert "dominator tree of depth 3" in note
        assert "pass(es)" in note
