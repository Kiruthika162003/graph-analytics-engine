from __future__ import annotations

import random
from itertools import combinations

import pytest

from mesh.connectedcomponents import ConnectedComponents
from mesh.errors import Invalid, Missing, Unreachable
from mesh.graph import Graph
from mesh.kruskal import Kruskal
from mesh.steiner import SteinerTree


def _brute_steiner(g: Graph, terminals: list[str]) -> float:
    # try every subset of non-terminals as waypoints; MST of the induced graph
    others = [n for n in g.nodes() if n not in terminals]
    best = float("inf")
    for size in range(len(others) + 1):
        for extra in combinations(others, size):
            keep = set(terminals) | set(extra)
            sub = Graph()
            for n in keep:
                sub.add_node(n)
            for u, v, w in g.edges():
                if u in keep and v in keep:
                    sub.add_edge(u, v, w)
            k = Kruskal(sub)
            if k.spans():
                best = min(best, k.total_weight())
    return best


def _star_of_three() -> Graph:
    # terminals a, b, c around a cheap center h; direct edges are dear
    g = Graph()
    for n in "abch":
        g.add_node(n)
    for t in "abc":
        g.add_edge(t, "h", 1)
    g.add_edge("a", "b", 5)
    g.add_edge("b", "c", 5)
    g.add_edge("a", "c", 5)
    return g


class TestTree:
    def test_it_borrows_the_cheap_center(self):
        st = SteinerTree(_star_of_three(), ["a", "b", "c"])
        assert st.weight() == 3
        assert st.borrowed() == {"h"}

    def test_the_result_connects_every_terminal(self):
        st = SteinerTree(_star_of_three(), ["a", "b", "c"])
        tree = Graph()
        for u, v, w in st.edges:
            tree.add_node(u)
            tree.add_node(v)
            tree.add_edge(u, v, w)
        cc = ConnectedComponents(tree)
        assert cc.component_of("a") >= {"a", "b", "c"}

    def test_two_terminals_give_their_shortest_path(self):
        g = _star_of_three()
        st = SteinerTree(g, ["a", "b"])
        assert st.weight() == 2  # a-h-b beats the direct 5

    def test_non_terminal_leaves_are_pruned(self):
        g = _star_of_three()
        g.add_node("dangling")
        g.add_edge("h", "dangling", 1)
        st = SteinerTree(g, ["a", "b", "c"])
        assert "dangling" not in st.borrowed()


class TestGuarantee:
    def test_the_weight_stays_within_twice_the_optimum(self):
        rng = random.Random(157)
        for _ in range(20):
            g = Graph()
            nodes = [str(i) for i in range(7)]
            for n in nodes:
                g.add_node(n)
            for a, b in combinations(nodes, 2):
                if rng.random() < 0.5:
                    g.add_edge(a, b, rng.randint(1, 9))
            terminals = ["0", "1", "2"]
            if not ConnectedComponents(g).is_connected():
                continue
            st = SteinerTree(g, terminals)
            optimum = _brute_steiner(g, terminals)
            assert optimum <= st.weight() <= 2 * optimum
            assert st.weight() >= st.lower_bound()


class TestRefusals:
    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            SteinerTree(Graph(directed=True), ["a", "b"])

    def test_fewer_than_two_terminals_is_refused(self):
        with pytest.raises(Invalid):
            SteinerTree(_star_of_three(), ["a"])

    def test_a_terminal_outside_the_graph_is_refused(self):
        with pytest.raises(Missing):
            SteinerTree(_star_of_three(), ["a", "ghost"])

    def test_unreachable_terminals_are_refused(self):
        # the closure raises Unreachable, the precise class; the module keeps
        # it and names the two terminals rather than recasting it as Invalid
        g = _star_of_three()
        g.add_node("island")
        with pytest.raises(Unreachable) as caught:
            SteinerTree(g, ["a", "island"])
        assert "'a' and 'island'" in str(caught.value)


class TestReport:
    def test_the_note_states_weight_and_borrowed(self):
        note = SteinerTree(_star_of_three(), ["a", "b", "c"]).note()
        assert "weight 3" in note
        assert "borrowing 1 waypoint(s)" in note
