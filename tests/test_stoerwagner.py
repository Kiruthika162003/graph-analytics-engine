from __future__ import annotations

import random
from itertools import combinations

import pytest

from mesh.errors import Invalid
from mesh.graph import Graph
from mesh.stoerwagner import StoerWagner


def _two_cliques_bridged(bridge_weight: float = 1.0) -> Graph:
    g = Graph()
    left = [f"l{i}" for i in range(4)]
    right = [f"r{i}" for i in range(4)]
    for n in left + right:
        g.add_node(n)
    for a, b in combinations(left, 2):
        g.add_edge(a, b, 5)
    for a, b in combinations(right, 2):
        g.add_edge(a, b, 5)
    g.add_edge("l0", "r0", bridge_weight)
    return g


def _brute_min_cut(g: Graph) -> float:
    nodes = g.nodes()
    best = float("inf")
    for mask in range(1, 1 << (len(nodes) - 1)):  # fix the last node's side
        side = {nodes[i] for i in range(len(nodes)) if mask >> i & 1}
        cut = sum(w for u, v, w in g.edges() if (u in side) != (v in side))
        best = min(best, cut)
    return best


class TestCut:
    def test_the_bridge_is_the_global_minimum_cut(self):
        sw = StoerWagner(_two_cliques_bridged())
        assert sw.weight == 1

    def test_the_partition_separates_the_two_cliques(self):
        sw = StoerWagner(_two_cliques_bridged())
        side, rest = sw.sides()
        left = {f"l{i}" for i in range(4)}
        assert left in (side, rest)

    def test_a_heavy_bridge_moves_the_seam_to_a_single_node(self):
        # bridge weight 20 exceeds any node's degree of 15 inside a 5-weight K4
        sw = StoerWagner(_two_cliques_bridged(bridge_weight=20))
        assert sw.weight == 15
        assert "around one poorly attached node" in sw.note()

    def test_a_single_edge_graph_cuts_that_edge(self):
        g = Graph()
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b", 7)
        assert StoerWagner(g).weight == 7

    def test_a_disconnected_graph_has_a_zero_cut(self):
        g = Graph()
        for n in "abcd":
            g.add_node(n)
        g.add_edge("a", "b")
        g.add_edge("c", "d")
        assert StoerWagner(g).weight == 0


class TestRefusals:
    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            StoerWagner(Graph(directed=True))

    def test_a_single_node_is_refused(self):
        g = Graph()
        g.add_node("a")
        with pytest.raises(Invalid):
            StoerWagner(g)


class TestAgainstBruteForce:
    def test_the_cut_weight_matches_trying_every_bipartition(self):
        rng = random.Random(149)
        for _ in range(30):
            g = Graph()
            nodes = [str(i) for i in range(7)]
            for n in nodes:
                g.add_node(n)
            for a, b in combinations(nodes, 2):
                if rng.random() < 0.5:
                    g.add_edge(a, b, rng.randint(1, 9))
            sw = StoerWagner(g)
            assert sw.weight == _brute_min_cut(g)
            # and the reported partition really has that cut weight
            side, _rest = sw.sides()
            realized = sum(w for u, v, w in g.edges() if (u in side) != (v in side))
            assert realized == sw.weight


class TestReport:
    def test_the_note_states_the_cut_and_the_seam(self):
        note = StoerWagner(_two_cliques_bridged()).note()
        assert "global minimum cut 1" in note
        assert "between groups" in note
