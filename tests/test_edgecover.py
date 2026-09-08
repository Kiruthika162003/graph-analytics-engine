from __future__ import annotations

import random
from itertools import combinations

import pytest

from mesh.edgecover import EdgeCover
from mesh.errors import Invalid
from mesh.factories import complete, cycle, path, star
from mesh.graph import Graph


def _brute_minimum(g: Graph) -> int:
    edges = [(u, v) for u, v, _w in g.edges()]
    nodes = set(g.nodes())
    for k in range(1, len(edges) + 1):
        for subset in combinations(edges, k):
            if {n for e in subset for n in e} == nodes:
                return k
    return 0


def _random_without_isolated(rng: random.Random, n: int) -> Graph:
    g = Graph()
    nodes = [str(i) for i in range(n)]
    for node in nodes:
        g.add_node(node)
    for a, b in combinations(nodes, 2):
        if rng.random() < 0.35:
            g.add_edge(a, b)
    for node in nodes:
        if g.degree(node) == 0:
            other = rng.choice([m for m in nodes if m != node])
            g.add_edge(node, other)
    return g


class TestCover:
    def test_a_star_needs_every_leaf_edge(self):
        ec = EdgeCover(star(4))
        assert len(ec.cover) == 4
        assert ec.covers_every_node()
        assert ec.gallai_holds()

    def test_an_even_cycle_is_covered_by_a_perfect_matching(self):
        ec = EdgeCover(cycle(6))
        assert len(ec.cover) == 3
        assert ec.matching_size() == 3

    def test_an_odd_cycle_needs_one_extra(self):
        ec = EdgeCover(cycle(5))
        assert len(ec.cover) == 3
        assert ec.covers_every_node()

    def test_a_path_of_five_needs_three(self):
        assert len(EdgeCover(path(5)).cover) == 3

    def test_a_complete_graph_needs_half_rounded_up(self):
        assert len(EdgeCover(complete(6)).cover) == 3
        assert len(EdgeCover(complete(7)).cover) == 4


class TestAgainstBruteForce:
    def test_the_construction_matches_the_smallest_subset_on_random_graphs(self):
        rng = random.Random(557)
        for _ in range(20):
            g = _random_without_isolated(rng, 7)
            ec = EdgeCover(g)
            assert ec.covers_every_node()
            assert ec.gallai_holds()
            assert len(ec.cover) == _brute_minimum(g)


class TestRefusal:
    def test_an_isolated_node_is_refused_by_name(self):
        g = path(3)
        g.add_node("alone")
        with pytest.raises(Invalid, match="'alone' has no edge"):
            EdgeCover(g)

    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            EdgeCover(Graph(directed=True))


class TestReport:
    def test_the_note_states_the_sizes_and_the_identity(self):
        note = EdgeCover(cycle(5)).note()
        assert "3 edge(s) cover 5 node(s) from a matching of 2; Gallai's identity holds" in note
