from __future__ import annotations

import random

import pytest

from mesh.errors import Invalid, Missing
from mesh.eulertour import EulerTour
from mesh.graph import Graph


def _tree() -> tuple[Graph, dict[str, float]]:
    #        r
    #      /   \
    #     a     b
    #    / \     \
    #   c   d     e
    g = Graph()
    for n in "rabcde":
        g.add_node(n)
    for u, v in [("r", "a"), ("r", "b"), ("a", "c"), ("a", "d"), ("b", "e")]:
        g.add_edge(u, v)
    values = {"r": 1.0, "a": 5.0, "b": 2.0, "c": 9.0, "d": 3.0, "e": 7.0}
    return g, values


def _random_tree(rng: random.Random, n: int) -> tuple[Graph, dict[str, float], dict[str, str]]:
    g = Graph()
    g.add_node("0")
    parent: dict[str, str] = {}
    for i in range(1, n):
        p = str(rng.randrange(i))
        g.add_node(str(i))
        g.add_edge(p, str(i))
        parent[str(i)] = p
    values = {node: float(rng.randint(0, 50)) for node in g.nodes()}
    return g, values, parent


def _descendants(parent: dict[str, str], node: str, nodes: list[str]) -> set[str]:
    out = {node}
    changed = True
    while changed:
        changed = False
        for n in nodes:
            if n in parent and parent[n] in out and n not in out:
                out.add(n)
                changed = True
    return out


class TestTour:
    def test_each_node_is_written_twice(self):
        g, values = _tree()
        et = EulerTour(g, "r", values)
        assert len(et.tour) == 2 * g.node_count()

    def test_a_subtree_is_a_contiguous_range(self):
        g, values = _tree()
        et = EulerTour(g, "r", values)
        assert et.subtree_size("a") == 3
        assert et.subtree_size("r") == 6
        assert et.subtree_size("c") == 1

    def test_ancestry_follows_interval_nesting(self):
        g, values = _tree()
        et = EulerTour(g, "r", values)
        assert et.is_ancestor("r", "e")
        assert et.is_ancestor("a", "d")
        assert not et.is_ancestor("a", "e")
        assert et.is_ancestor("c", "c")

    def test_subtree_max_reads_the_range(self):
        g, values = _tree()
        et = EulerTour(g, "r", values)
        assert et.subtree_max("a") == 9.0
        assert et.subtree_max("b") == 7.0

    def test_an_update_is_seen_by_every_containing_subtree(self):
        g, values = _tree()
        et = EulerTour(g, "r", values)
        et.update("d", 42.0)
        assert et.subtree_max("a") == 42.0
        assert et.subtree_max("r") == 42.0
        assert et.subtree_max("b") == 7.0


class TestRefusals:
    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            EulerTour(Graph(directed=True), "r", {})

    def test_a_missing_root_is_refused(self):
        g, values = _tree()
        with pytest.raises(Missing):
            EulerTour(g, "ghost", values)

    def test_a_cycle_is_refused(self):
        g, values = _tree()
        g.add_edge("c", "e")
        with pytest.raises(Invalid):
            EulerTour(g, "r", values)

    def test_a_missing_query_node_is_refused(self):
        g, values = _tree()
        with pytest.raises(Missing):
            EulerTour(g, "r", values).subtree_size("ghost")


class TestAgainstDescendantSets:
    def test_size_max_and_ancestry_match_explicit_descendant_sets(self):
        rng = random.Random(241)
        for _ in range(20):
            g, values, parent = _random_tree(rng, rng.randint(1, 40))
            et = EulerTour(g, "0", values)
            nodes = g.nodes()
            for node in nodes:
                desc = _descendants(parent, node, nodes)
                assert et.subtree_size(node) == len(desc)
                assert et.subtree_max(node) == max(values[d] for d in desc)
                for other in nodes:
                    assert et.is_ancestor(node, other) == (other in desc)


class TestReport:
    def test_the_note_states_the_tour_length(self):
        g, values = _tree()
        assert "tour of length 12 over 6 node(s)" in EulerTour(g, "r", values).note()
