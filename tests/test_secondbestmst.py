from __future__ import annotations

import random
from itertools import combinations, pairwise

import pytest

from mesh.errors import Invalid
from mesh.graph import Graph
from mesh.secondbestmst import SecondBestMST
from mesh.unionfind import UnionFind


def _all_spanning_weights(g: Graph) -> list[float]:
    nodes = g.nodes()
    weights = []
    for subset in combinations(g.edges(), len(nodes) - 1):
        uf = UnionFind()
        for n in nodes:
            uf.add(n)
        if all(uf.union(u, v) for u, v, _w in subset):
            weights.append(sum(w for _u, _v, w in subset))
    return sorted(weights)


def _square_with_diagonals() -> Graph:
    g = Graph()
    for n in "abcd":
        g.add_node(n)
    g.add_edge("a", "b", 1)
    g.add_edge("b", "c", 2)
    g.add_edge("c", "d", 3)
    g.add_edge("d", "a", 4)
    g.add_edge("a", "c", 5)
    g.add_edge("b", "d", 6)
    return g


class TestSecondBest:
    def test_the_runner_up_beats_every_other_tree_but_the_best(self):
        g = _square_with_diagonals()
        sb = SecondBestMST(g)
        weights = _all_spanning_weights(g)
        assert sb.best == weights[0]
        assert sb.second == weights[1]

    def test_the_swap_is_one_edge_in_and_one_out(self):
        sb = SecondBestMST(_square_with_diagonals())
        # best is a-b, b-c, c-d = 6; second swaps d-a (4) for c-d (3) = 7
        assert sb.second == 7
        assert sb.added is not None and sb.added[2] == 4
        assert sb.dropped_weight == 3
        assert sb.unique_minimum()

    def test_a_tie_is_reported_as_a_non_unique_minimum(self):
        g = Graph()
        for n in "abc":
            g.add_node(n)
        g.add_edge("a", "b", 1)
        g.add_edge("b", "c", 1)
        g.add_edge("c", "a", 1)
        sb = SecondBestMST(g)
        assert not sb.unique_minimum()
        assert "not unique" in sb.note()


class TestRefusals:
    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            SecondBestMST(Graph(directed=True))

    def test_a_tree_has_no_second_best(self):
        g = Graph()
        for n in "abc":
            g.add_node(n)
        g.add_edge("a", "b", 1)
        g.add_edge("b", "c", 2)
        with pytest.raises(Invalid):
            SecondBestMST(g)

    def test_a_disconnected_graph_is_refused(self):
        g = Graph()
        for n in "abcd":
            g.add_node(n)
        g.add_edge("a", "b", 1)
        g.add_edge("c", "d", 1)
        with pytest.raises(Invalid):
            SecondBestMST(g)


class TestAgainstEnumeration:
    def test_the_second_weight_matches_sorting_every_spanning_tree(self):
        rng = random.Random(347)
        for _ in range(25):
            g = Graph()
            nodes = [str(i) for i in range(6)]
            for n in nodes:
                g.add_node(n)
            shuffled = nodes[:]
            rng.shuffle(shuffled)
            for a, b in pairwise(shuffled):
                g.add_edge(a, b, rng.randint(1, 12))
            for _ in range(4):
                a, b = rng.sample(nodes, 2)
                if not g.has_edge(a, b):
                    g.add_edge(a, b, rng.randint(1, 12))
            if g.edge_count() == len(nodes) - 1:
                continue
            weights = _all_spanning_weights(g)
            sb = SecondBestMST(g)
            assert sb.best == weights[0]
            assert sb.second == weights[1]


class TestReport:
    def test_the_note_states_the_margin(self):
        assert "a margin of 1" in SecondBestMST(_square_with_diagonals()).note()
