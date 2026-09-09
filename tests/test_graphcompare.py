from __future__ import annotations

import random
from itertools import combinations

from mesh.factories import complete, cycle, path, star
from mesh.graph import Graph
from mesh.graphcompare import GraphCompare


def _relabel(g: Graph, seed: int) -> Graph:
    rng = random.Random(seed)
    names = g.nodes()
    shuffled = list(names)
    rng.shuffle(shuffled)
    mapping = dict(zip(names, shuffled, strict=True))
    h = Graph()
    for n in shuffled:
        h.add_node(n)
    for u, v, _w in g.edges():
        h.add_edge(mapping[u], mapping[v])
    return h


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


class TestIsomorphicPairs:
    def test_a_relabeled_copy_reads_the_same_on_every_scale(self):
        g = _random_graph(953, 6, 0.5)
        gc = GraphCompare(g, _relabel(g, 1))
        assert gc.isomorphic
        assert gc.same_digest
        assert gc.edit == 0
        assert gc.common == 6
        assert abs(gc.kernel - 1.0) < 1e-9
        assert gc.degree_gap == 0
        assert gc.consistent()


class TestDifferentPairs:
    def test_a_path_and_a_star_differ_everywhere_but_stay_consistent(self):
        gc = GraphCompare(path(4), star(3))
        assert not gc.isomorphic
        assert not gc.same_digest
        assert gc.edit == 2
        assert gc.common == 3
        assert gc.kernel < 1.0
        assert gc.degree_gap == 2
        assert gc.consistent()

    def test_cospectral_but_different_shapes_are_told_apart_by_the_digest(self):
        square = cycle(4, prefix="c")
        square.add_node("lone")
        gc = GraphCompare(square, star(4, prefix="s"))
        assert not gc.same_digest
        assert not gc.isomorphic
        assert gc.consistent()

    def test_consistency_holds_on_random_pairs(self):
        for seed in range(957, 969):
            a = _random_graph(seed, 6, 0.5)
            b = _random_graph(seed + 50, 6, 0.5)
            assert GraphCompare(a, b).consistent()


class TestLimits:
    def test_large_pairs_skip_the_exact_readings_and_say_so(self):
        gc = GraphCompare(cycle(9), cycle(9, prefix="r"))
        assert gc.edit is None
        assert gc.common is None
        assert gc.isomorphic
        assert "edit distance skipped" in gc.note()
        assert "common subgraph skipped" in gc.note()
        assert gc.consistent()

    def test_the_note_lists_every_reading(self):
        note = GraphCompare(complete(3), path(3)).note()
        assert "digests differ, not isomorphic, edit distance 1, common subgraph 2" in note
        assert "walk similarity" in note
        assert "degree gap 2" in note
