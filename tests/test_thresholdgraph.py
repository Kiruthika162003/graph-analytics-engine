from __future__ import annotations

import random
from itertools import combinations

import pytest

from mesh.errors import Invalid
from mesh.factories import complete, cycle, path, star
from mesh.graph import Graph
from mesh.thresholdgraph import DOMINATING, ISOLATED, ThresholdGraph


def _grow(rng: random.Random, n: int) -> Graph:
    g = Graph()
    g.add_node("0")
    for i in range(1, n):
        name = str(i)
        earlier = g.nodes()
        g.add_node(name)
        if rng.random() < 0.5:
            for other in earlier:
                g.add_edge(name, other)
    return g


class TestVerdict:
    def test_a_star_grows_as_leaves_then_a_dominating_hub(self):
        t = ThresholdGraph(star(3))
        assert t.is_threshold
        assert t.sequence[-1] == ("0", DOMINATING)
        assert all(kind == ISOLATED for _n, kind in t.sequence[1:-1])

    def test_complete_and_edgeless_graphs_are_threshold(self):
        assert ThresholdGraph(complete(4)).is_threshold
        g = Graph()
        for n in "abc":
            g.add_node(n)
        assert ThresholdGraph(g).is_threshold

    def test_the_three_forbidden_shapes_are_refused(self):
        assert not ThresholdGraph(cycle(4)).is_threshold
        assert not ThresholdGraph(path(4)).is_threshold
        two_edges = Graph()
        for n in "abcd":
            two_edges.add_node(n)
        two_edges.add_edge("a", "b")
        two_edges.add_edge("c", "d")
        t = ThresholdGraph(two_edges)
        assert not t.is_threshold
        assert t.stuck == {"a", "b", "c", "d"}


class TestNumbers:
    def test_sequence_numbers_match_the_other_modules_on_grown_graphs(self):
        rng = random.Random(471)
        for _ in range(25):
            t = ThresholdGraph(_grow(rng, 8))
            assert t.is_threshold
            assert t.numbers_agree()

    def test_an_empty_graph_has_zero_for_both_numbers(self):
        t = ThresholdGraph(Graph())
        assert t.clique_number() == 0
        assert t.independence_number() == 0

    def test_the_numbers_are_refused_off_a_threshold_graph(self):
        with pytest.raises(Invalid):
            ThresholdGraph(cycle(4)).clique_number()


class TestIdentity:
    def test_threshold_equals_split_and_cograph_on_random_graphs(self):
        rng = random.Random(473)
        for _ in range(30):
            g = Graph()
            nodes = [str(i) for i in range(7)]
            for n in nodes:
                g.add_node(n)
            for a, b in combinations(nodes, 2):
                if rng.random() < 0.5:
                    g.add_edge(a, b)
            assert ThresholdGraph(g).matches_split_and_cograph()


class TestRefusal:
    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            ThresholdGraph(Graph(directed=True))


class TestReport:
    def test_the_note_states_the_creation_sequence(self):
        # the guess was [1:i, 2:i, 0:d]; peeling takes the smallest name first, so the
        # reversed order lists the later-peeled leaf before the earlier one
        note = ThresholdGraph(star(2)).note()
        assert "grown as [2:i, 1:i, 0:d]" in note
        assert "clique number 2, independence number 2" in note

    def test_the_note_counts_the_stuck_nodes(self):
        assert "4 node(s) remain" in ThresholdGraph(cycle(4)).note()
