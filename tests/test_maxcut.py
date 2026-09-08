from __future__ import annotations

import random
from itertools import combinations

import pytest

from mesh.errors import Invalid
from mesh.graph import Graph
from mesh.maxcut import MaxCut


def _brute_max_cut(g: Graph) -> float:
    nodes = g.nodes()
    best = 0.0
    for mask in range(1 << (len(nodes) - 1)):
        side = {nodes[i] for i in range(len(nodes)) if mask >> i & 1}
        cut = sum(w for u, v, w in g.edges() if (u in side) != (v in side))
        best = max(best, cut)
    return best


def _cycle(k: int) -> Graph:
    g = Graph()
    nodes = [str(i) for i in range(k)]
    for n in nodes:
        g.add_node(n)
    for i in range(k):
        g.add_edge(nodes[i], nodes[(i + 1) % k])
    return g


class TestCut:
    def test_a_bipartite_graph_can_cut_every_edge(self):
        mc = MaxCut(_cycle(6), seed=1)
        assert mc.cut_weight() == 6

    def test_an_odd_cycle_cuts_all_but_one(self):
        mc = MaxCut(_cycle(5), seed=2)
        assert mc.cut_weight() == 4

    def test_the_cut_never_falls_below_half_the_total(self):
        rng = random.Random(181)
        for seed in range(20):
            g = Graph()
            nodes = [str(i) for i in range(8)]
            for n in nodes:
                g.add_node(n)
            for a, b in combinations(nodes, 2):
                if rng.random() < 0.5:
                    g.add_edge(a, b, rng.randint(1, 5))
            if g.edge_count() == 0:
                continue
            mc = MaxCut(g, seed=seed)
            assert mc.cut_weight() >= mc.guarantee_bound()

    def test_no_single_move_improves_the_result(self):
        g = _cycle(7)
        g.add_edge("0", "3")
        mc = MaxCut(g, seed=3)
        for node in g.nodes():
            assert mc.gain(node) <= 0

    def test_a_single_start_can_stall_below_the_full_bipartite_cut(self):
        # first guess: local search always finds a bipartite graph's full cut.
        # Refuted: on a six-cycle a split whose same-side edges form a
        # matching gives every node gain zero and stops at four of six.
        stalled = [MaxCut(_cycle(6), seed=s, restarts=1).cut_weight() for s in range(30)]
        assert min(stalled) < 6
        assert all(w >= 3 for w in stalled)  # the half guarantee still holds

    def test_the_sides_partition_every_node(self):
        mc = MaxCut(_cycle(6), seed=4)
        a, b = mc.sides()
        assert a | b == set(_cycle(6).nodes())
        assert not (a & b)


class TestRefusals:
    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            MaxCut(Graph(directed=True))

    def test_a_single_node_is_refused(self):
        g = Graph()
        g.add_node("a")
        with pytest.raises(Invalid):
            MaxCut(g)


class TestAgainstBruteForce:
    def test_the_local_search_is_measured_against_the_true_maximum(self):
        rng = random.Random(191)
        gaps = []
        for seed in range(25):
            g = Graph()
            nodes = [str(i) for i in range(7)]
            for n in nodes:
                g.add_node(n)
            for a, b in combinations(nodes, 2):
                if rng.random() < 0.5:
                    g.add_edge(a, b)
            if g.edge_count() == 0:
                continue
            found = MaxCut(g, seed=seed).cut_weight()
            best = _brute_max_cut(g)
            assert found <= best
            assert found >= best / 2
            gaps.append(best - found)
        # the heuristic reaches the optimum on most small instances
        assert gaps.count(0) >= len(gaps) // 2


class TestReport:
    def test_the_note_states_the_cut_share(self):
        note = MaxCut(_cycle(6), seed=5).note()
        assert "cut 6 of total 6 (100%)" in note
