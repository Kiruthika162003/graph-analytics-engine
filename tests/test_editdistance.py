from __future__ import annotations

import random
from itertools import combinations

import pytest

from mesh.editdistance import EditDistance
from mesh.errors import Invalid
from mesh.factories import complete, cycle, path, star
from mesh.graph import Graph


def _random_graph(seed: int, n: int, p: float) -> Graph:
    rng = random.Random(seed)
    g = Graph()
    nodes = [f"{seed}_{i}" for i in range(n)]
    for node in nodes:
        g.add_node(node)
    for a, b in combinations(nodes, 2):
        if rng.random() < p:
            g.add_edge(a, b)
    return g


class TestSmallDistances:
    def test_a_graph_to_itself_is_zero(self):
        assert EditDistance(cycle(5), cycle(5)).distance == 0

    def test_a_path_of_three_to_a_triangle_is_one_edge(self):
        assert EditDistance(path(3), cycle(3)).distance == 1

    def test_a_relabeled_copy_is_zero(self):
        g = Graph()
        for n in "xyz":
            g.add_node(n)
        g.add_edge("x", "y")
        g.add_edge("y", "z")
        ed = EditDistance(path(3), g)
        assert ed.distance == 0
        assert len(ed.matched_pairs()) == 3

    def test_to_the_empty_graph_costs_nodes_plus_edges(self):
        assert EditDistance(star(3), Graph()).distance == 4 + 3
        assert EditDistance(Graph(), star(3)).distance == 4 + 3

    def test_adding_a_pendant_node_costs_two(self):
        g = cycle(4)
        g.add_node("tail")
        g.add_edge("0", "tail")
        assert EditDistance(cycle(4), g).distance == 2

    def test_a_square_to_a_complete_four_costs_the_two_diagonals(self):
        assert EditDistance(cycle(4), complete(4)).distance == 2


class TestSymmetry:
    def test_the_distance_is_the_same_both_ways_on_random_pairs(self):
        for seed in range(797, 805):
            a = _random_graph(seed, 5, 0.5)
            b = _random_graph(seed + 100, 5, 0.5)
            assert EditDistance(a, b).distance == EditDistance(b, a).distance

    def test_the_triangle_inequality_holds_on_a_few_triples(self):
        shapes = [path(4), cycle(4), star(3), complete(4)]
        for x in shapes:
            for y in shapes:
                for z in shapes:
                    d = EditDistance
                    assert d(x, z).distance <= d(x, y).distance + d(y, z).distance


class TestDirected:
    def test_reversing_one_arc_costs_two_in_a_directed_graph(self):
        a = Graph(directed=True)
        b = Graph(directed=True)
        for g in (a, b):
            g.add_node("p")
            g.add_node("q")
        a.add_edge("p", "q")
        b.add_edge("q", "p")
        # the search can relabel p and q, so a reversed arc costs nothing
        assert EditDistance(a, b).distance == 0
        c = Graph(directed=True)
        for n in "pqr":
            c.add_node(n)
        c.add_edge("p", "q")
        c.add_edge("q", "r")
        d = Graph(directed=True)
        for n in "pqr":
            d.add_node(n)
        d.add_edge("p", "q")
        d.add_edge("r", "q")
        assert EditDistance(c, d).distance == 2


class TestRefusal:
    def test_mixed_direction_and_large_graphs_are_refused(self):
        with pytest.raises(Invalid):
            EditDistance(Graph(), Graph(directed=True))
        with pytest.raises(Invalid):
            EditDistance(path(9), path(2))


class TestReport:
    def test_the_note_states_the_distance_and_the_matches(self):
        note = EditDistance(path(3), cycle(3)).note()
        assert note.startswith("edit distance 1 with 3 node(s) matched")
