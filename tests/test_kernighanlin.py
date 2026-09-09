from __future__ import annotations

import random
from itertools import combinations

import pytest

from mesh.errors import Invalid
from mesh.factories import cycle, path
from mesh.graph import Graph
from mesh.graphproduct import GraphProduct
from mesh.kernighanlin import KernighanLin


def _two_cliques() -> Graph:
    g = Graph()
    for group in ("abcd", "wxyz"):
        for n in group:
            g.add_node(n)
        for a, b in combinations(group, 2):
            g.add_edge(a, b)
    g.add_edge("d", "w")
    return g


def _random_graph(seed: int, n: int, p: float) -> Graph:
    rng = random.Random(seed)
    g = Graph()
    nodes = [str(i) for i in range(n)]
    for node in nodes:
        g.add_node(node)
    for a, b in combinations(nodes, 2):
        if rng.random() < p:
            g.add_edge(a, b, float(rng.randint(1, 4)))
    return g


class TestShapes:
    def test_two_cliques_separate_at_the_bridge_from_a_mixed_start(self):
        kl = KernighanLin(_two_cliques(), left={"a", "b", "w", "x"})
        assert kl.initial_cut == 9
        assert kl.cut_weight() == 1
        assert kl.left in ({*"abcd"}, {*"wxyz"})
        assert kl.balanced()

    def test_a_ladder_cuts_at_a_rung(self):
        ladder = GraphProduct(path(2), path(4)).cartesian()
        kl = KernighanLin(ladder, left={"0,0", "1,0", "0,3", "1,3"})
        assert kl.cut_weight() == 2

    def test_an_even_cycle_cuts_in_two(self):
        kl = KernighanLin(cycle(8), left={"0", "2", "4", "6"})
        assert kl.cut_weight() == 2


class TestAgainstExact:
    def test_the_heuristic_never_beats_the_exact_minimum_and_stays_balanced(self):
        for seed in range(761, 773):
            g = _random_graph(seed, 8, 0.45)
            kl = KernighanLin(g)
            assert kl.balanced()
            assert kl.cut_weight() >= kl.exact_minimum() - 1e-9
            assert kl.cut_weight() <= kl.initial_cut + 1e-9

    def test_the_heuristic_reaches_the_exact_minimum_often(self):
        reached = 0
        for seed in range(773, 793):
            g = _random_graph(seed, 8, 0.45)
            kl = KernighanLin(g)
            if kl.cut_weight() == kl.exact_minimum():
                reached += 1
        assert reached >= 12


class TestRefusal:
    def test_odd_counts_directed_graphs_and_bad_starts_are_refused(self):
        with pytest.raises(Invalid):
            KernighanLin(path(5))
        with pytest.raises(Invalid):
            KernighanLin(Graph(directed=True))
        with pytest.raises(Invalid):
            KernighanLin(path(4), left={"0"})
        with pytest.raises(Invalid):
            KernighanLin(path(4), left={"0", "zz"})

    def test_the_exact_search_refuses_large_graphs(self):
        with pytest.raises(Invalid):
            KernighanLin(path(16)).exact_minimum()


class TestReport:
    def test_the_note_states_the_cut_before_and_after(self):
        note = KernighanLin(_two_cliques(), left={"a", "b", "w", "x"}).note()
        assert note.startswith("cut 9 to 1 in")
        assert "sides" in note
