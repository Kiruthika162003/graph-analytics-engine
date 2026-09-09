from __future__ import annotations

import random
from itertools import combinations

import pytest

from mesh.errors import Invalid
from mesh.factories import complete, cycle, path, star
from mesh.graph import Graph
from mesh.walkcount import WalkCount


def _random_graph(seed: int, n: int, p: float) -> Graph:
    rng = random.Random(seed)
    g = Graph()
    nodes = [str(i) for i in range(n)]
    for node in nodes:
        g.add_node(node)
    for a, b in combinations(nodes, 2):
        if rng.random() < p:
            g.add_edge(a, b)
    return g


class TestSmallLengths:
    def test_length_zero_and_one_read_the_identity_and_the_adjacency(self):
        wc = WalkCount(path(3))
        assert wc.walks("0", "0", 0) == 1
        assert wc.walks("0", "1", 0) == 0
        assert wc.walks("0", "1", 1) == 1
        assert wc.walks("0", "2", 1) == 0

    def test_length_two_between_distinct_nodes_counts_common_neighbors(self):
        for seed in range(1069, 1075):
            g = _random_graph(seed, 8, 0.5)
            wc = WalkCount(g)
            for a, b in combinations(g.nodes(), 2):
                assert wc.walks(a, b, 2) == wc.common_neighbors(a, b)

    def test_closed_walks_of_three_are_twice_the_triangles_at_the_node(self):
        g = _random_graph(1087, 8, 0.5)
        wc = WalkCount(g)
        for n in g.nodes():
            assert wc.closed(n, 3) == 2 * wc.triangles_at(n)

    def test_closed_walks_of_two_are_the_degree(self):
        wc = WalkCount(star(4))
        assert wc.closed("0", 2) == 4
        assert wc.closed("1", 2) == 1


class TestTraces:
    def test_traces_match_the_spectrum_at_several_lengths(self):
        for g in (cycle(6), complete(5), _random_graph(1091, 7, 0.5)):
            wc = WalkCount(g)
            for k in (2, 3, 4, 5):
                assert wc.matches_spectrum(k)

    def test_a_five_cycle_closes_a_walk_of_five_only_by_going_round(self):
        # an odd closed walk on an odd cycle must wind around; at length five that is
        # once each way, so two per node and ten in the trace
        wc = WalkCount(cycle(5))
        assert wc.closed("0", 5) == 2
        assert wc.trace(5) == 10

    def test_powers_are_kept_between_questions(self):
        wc = WalkCount(cycle(4))
        wc.trace(6)
        assert len(wc.powers) == 7
        wc.trace(3)
        assert len(wc.powers) == 7


class TestDirected:
    def test_arcs_are_followed_one_way_and_a_directed_triangle_has_three_closed_walks(self):
        g = Graph(directed=True)
        for n in "abc":
            g.add_node(n)
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        g.add_edge("c", "a")
        wc = WalkCount(g)
        assert wc.walks("a", "c", 2) == 1
        assert wc.walks("c", "a", 2) == 0
        assert wc.trace(3) == 3
        with pytest.raises(Invalid):
            wc.matches_spectrum(3)


class TestRefusal:
    def test_negative_lengths_and_unknown_names_are_refused(self):
        wc = WalkCount(path(2))
        with pytest.raises(Invalid):
            wc.power(-1)
        with pytest.raises(Invalid):
            wc.walks("0", "zz", 1)


class TestReport:
    def test_the_note_states_the_trace(self):
        assert WalkCount(cycle(3)).note(3) == "6 closed walk(s) of length 3 over 3 node(s)"
