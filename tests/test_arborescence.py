from __future__ import annotations

import random
from itertools import product

import pytest

from mesh.arborescence import Arborescence
from mesh.bfs import BFS
from mesh.errors import Invalid, Missing
from mesh.graph import Graph


def _brute_arborescence(g: Graph, root: str) -> float:
    # every non-root node picks one incoming edge; keep the combos that form
    # a tree reaching everything from the root, take the cheapest
    others = [n for n in g.nodes() if n != root]
    choices = []
    for v in others:
        incoming = [(u, w) for u, x, w in g.edges() if x == v]
        if not incoming:
            return float("inf")
        choices.append(incoming)
    best = float("inf")
    for combo in product(*choices):
        tree = Graph(directed=True)
        for n in g.nodes():
            tree.add_node(n)
        for v, (u, w) in zip(others, combo, strict=True):
            tree.add_edge(u, v, w)
        if len(BFS(tree, root).reachable_nodes()) == g.node_count():
            best = min(best, sum(w for _u, w in combo))
    return best


def _cycle_trap() -> Graph:
    # r -> a (5), a -> b (1), b -> a (1), r -> b (5): cheapest-in picks the
    # a<->b cycle, which no arborescence can contain
    g = Graph(directed=True)
    for n in "rab":
        g.add_node(n)
    g.add_edge("r", "a", 5)
    g.add_edge("a", "b", 1)
    g.add_edge("b", "a", 1)
    g.add_edge("r", "b", 5)
    return g


class TestArborescence:
    def test_a_simple_tree_costs_its_edges(self):
        g = Graph(directed=True)
        for n in "rab":
            g.add_node(n)
        g.add_edge("r", "a", 2)
        g.add_edge("a", "b", 3)
        assert Arborescence(g, "r").total == 5

    def test_a_cycle_among_cheapest_in_edges_is_broken_correctly(self):
        # keep one of the 1-edges and enter the cycle once at cost 5: total 6
        a = Arborescence(_cycle_trap(), "r")
        assert a.total == 6
        assert a.lower_bound == 2
        assert a.cycle_penalty() == 4

    def test_the_lower_bound_is_met_when_cheapest_edges_form_a_tree(self):
        g = Graph(directed=True)
        for n in "rabc":
            g.add_node(n)
        g.add_edge("r", "a", 1)
        g.add_edge("r", "b", 1)
        g.add_edge("a", "c", 1)
        g.add_edge("b", "c", 4)
        a = Arborescence(g, "r")
        assert a.total == a.lower_bound == 3


class TestRefusals:
    def test_an_undirected_graph_is_refused(self):
        with pytest.raises(Invalid):
            Arborescence(Graph(), "r")

    def test_a_missing_root_is_refused(self):
        g = Graph(directed=True)
        g.add_node("r")
        with pytest.raises(Missing):
            Arborescence(g, "ghost")

    def test_a_root_that_cannot_reach_everything_is_refused(self):
        g = _cycle_trap()
        g.add_node("island")
        with pytest.raises(Invalid):
            Arborescence(g, "r")


class TestAgainstBruteForce:
    def test_the_total_matches_trying_every_in_edge_choice(self):
        rng = random.Random(199)
        checked = 0
        for _ in range(40):
            g = Graph(directed=True)
            nodes = [str(i) for i in range(5)]
            for n in nodes:
                g.add_node(n)
            for u in nodes:
                for v in nodes:
                    if v not in (u, "0") and rng.random() < 0.5:
                        g.add_edge(u, v, rng.randint(1, 9))
            if len(BFS(g, "0").reachable_nodes()) != 5:
                continue
            checked += 1
            assert Arborescence(g, "0").total == _brute_arborescence(g, "0")
        assert checked > 10


class TestReport:
    def test_the_note_states_the_cycle_penalty(self):
        note = Arborescence(_cycle_trap(), "r").note()
        assert "costs 6" in note
        assert "gap of 4" in note
