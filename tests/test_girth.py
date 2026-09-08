from __future__ import annotations

import math
import random
from itertools import combinations

import pytest

from mesh.errors import Invalid
from mesh.girth import Girth
from mesh.graph import Graph


def _cycle(k: int) -> Graph:
    g = Graph()
    nodes = [str(i) for i in range(k)]
    for n in nodes:
        g.add_node(n)
    for i in range(k):
        g.add_edge(nodes[i], nodes[(i + 1) % k])
    return g


def _brute_girth(g: Graph) -> float:
    # shortest cycle by trying every simple cycle via DFS enumeration
    best = math.inf
    nodes = g.nodes()

    def walk(start: str, node: str, seen: list[str]) -> None:
        nonlocal best
        for nbr in g.neighbors(node):
            if nbr == start and len(seen) >= 3:
                best = min(best, len(seen))
            elif nbr not in seen and nbr > start:
                walk(start, nbr, [*seen, nbr])

    for s in nodes:
        walk(s, s, [s])
    return best


class TestGirth:
    def test_a_triangle_has_girth_three(self):
        assert Girth(_cycle(3)).girth == 3

    def test_a_pentagon_has_girth_five(self):
        assert Girth(_cycle(5)).girth == 5

    def test_a_tree_is_acyclic_with_infinite_girth(self):
        g = Graph()
        for n in "abcd":
            g.add_node(n)
        for u, v in [("a", "b"), ("b", "c"), ("b", "d")]:
            g.add_edge(u, v)
        gi = Girth(g)
        assert gi.is_acyclic()
        assert gi.girth == math.inf

    def test_the_shortest_cycle_wins_over_a_longer_one(self):
        # a hexagon with one chord making a quadrilateral
        g = _cycle(6)
        g.add_edge("0", "3")
        assert Girth(g).girth == 4

    def test_the_reported_cycle_has_the_girth_length_and_real_edges(self):
        g = _cycle(6)
        g.add_edge("0", "3")
        gi = Girth(g)
        cycle = gi.cycle
        assert len(cycle) == gi.girth
        for i in range(len(cycle)):
            assert g.has_edge(cycle[i], cycle[(i + 1) % len(cycle)])


class TestRefusal:
    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            Girth(Graph(directed=True))


class TestAgainstBruteForce:
    def test_it_matches_enumerating_every_simple_cycle(self):
        rng = random.Random(91)
        for _ in range(30):
            g = Graph()
            nodes = [str(i) for i in range(8)]
            for n in nodes:
                g.add_node(n)
            for a, b in combinations(nodes, 2):
                if rng.random() < 0.25:
                    g.add_edge(a, b)
            assert Girth(g).girth == _brute_girth(g)


class TestReport:
    def test_the_note_states_the_girth(self):
        assert "girth 5" in Girth(_cycle(5)).note()

    def test_the_note_says_acyclic_for_a_tree(self):
        g = Graph()
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b")
        assert "acyclic" in Girth(g).note()
