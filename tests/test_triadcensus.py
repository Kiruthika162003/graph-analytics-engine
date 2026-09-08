from __future__ import annotations

import random
from itertools import permutations
from math import comb

from mesh.factories import cycle, path
from mesh.graph import Graph
from mesh.triadcensus import TRIAD_NAMES, TriadCensus


def _digraph(arcs: list[tuple[str, str]], nodes: str = "abc") -> Graph:
    g = Graph(directed=True)
    for n in nodes:
        g.add_node(n)
    for u, v in arcs:
        g.add_edge(u, v)
    return g


class TestSingleTriads:
    def test_the_digit_only_shapes(self):
        assert TriadCensus(_digraph([])).counts["003"] == 1
        assert TriadCensus(_digraph([("a", "b")])).counts["012"] == 1
        assert TriadCensus(_digraph([("a", "b"), ("b", "a")])).counts["102"] == 1
        both = [("a", "b"), ("b", "a"), ("b", "c"), ("c", "b")]
        assert TriadCensus(_digraph(both)).counts["201"] == 1
        assert TriadCensus(_digraph([*both, ("a", "c")])).counts["210"] == 1
        every = [(x, y) for x, y in permutations("abc", 2)]
        assert TriadCensus(_digraph(every)).counts["300"] == 1

    def test_the_021_shapes_are_down_up_and_chain(self):
        assert TriadCensus(_digraph([("a", "b"), ("a", "c")])).counts["021D"] == 1
        assert TriadCensus(_digraph([("a", "b"), ("c", "b")])).counts["021U"] == 1
        assert TriadCensus(_digraph([("a", "b"), ("b", "c")])).counts["021C"] == 1

    def test_the_111_shapes_depend_on_which_way_the_free_arc_points(self):
        pair = [("a", "b"), ("b", "a")]
        assert TriadCensus(_digraph([*pair, ("c", "b")])).counts["111D"] == 1
        assert TriadCensus(_digraph([*pair, ("b", "c")])).counts["111U"] == 1

    def test_the_030_shapes_are_transitive_and_cyclic(self):
        assert TriadCensus(_digraph([("a", "b"), ("b", "c"), ("a", "c")])).counts["030T"] == 1
        assert TriadCensus(_digraph([("a", "b"), ("b", "c"), ("c", "a")])).counts["030C"] == 1

    def test_the_120_shapes_follow_the_021_letters(self):
        pair = [("b", "c"), ("c", "b")]
        assert TriadCensus(_digraph([*pair, ("a", "b"), ("a", "c")])).counts["120D"] == 1
        assert TriadCensus(_digraph([*pair, ("b", "a"), ("c", "a")])).counts["120U"] == 1
        pair = [("a", "c"), ("c", "a")]
        assert TriadCensus(_digraph([*pair, ("a", "b"), ("b", "c")])).counts["120C"] == 1


class TestWholeGraphs:
    def test_a_transitive_tournament_is_all_030t(self):
        names = "abcde"
        arcs = [(x, y) for i, x in enumerate(names) for y in names[i + 1 :]]
        tc = TriadCensus(_digraph(arcs, names))
        assert tc.counts["030T"] == comb(5, 3)
        assert tc.transitivity_share() == 1.0

    def test_counts_sum_to_n_choose_three_on_random_digraphs(self):
        rng = random.Random(601)
        for _ in range(10):
            g = Graph(directed=True)
            names = [str(i) for i in range(8)]
            for n in names:
                g.add_node(n)
            for a, b in permutations(names, 2):
                if rng.random() < 0.3:
                    g.add_edge(a, b)
            tc = TriadCensus(g)
            assert tc.sums_to_choose_three()
            assert set(tc.counts) == set(TRIAD_NAMES)

    def test_an_undirected_graph_lands_in_the_four_mutual_classes(self):
        tc = TriadCensus(cycle(5))
        assert tc.counts["102"] == 5
        assert tc.counts["201"] == 5
        assert tc.counts["003"] == 0
        assert sum(tc.counts[k] for k in ("003", "102", "201", "300")) == comb(5, 3)

    def test_a_directed_path_is_chains_and_singles(self):
        g = Graph(directed=True)
        for n in "abcd":
            g.add_node(n)
        for u, v in zip("abc", "bcd", strict=True):
            g.add_edge(u, v)
        tc = TriadCensus(g)
        assert tc.counts["021C"] == 2
        assert tc.counts["012"] == 2


class TestReport:
    def test_the_note_lists_classes_largest_first(self):
        note = TriadCensus(path(4)).note()
        assert note.startswith("4 triple(s): ")
        assert "102 x2" in note
        assert "201 x2" in note
        assert TriadCensus(Graph()).note() == "0 triple(s): no triples"
