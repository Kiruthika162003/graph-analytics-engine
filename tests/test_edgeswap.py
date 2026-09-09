from __future__ import annotations

import random
from itertools import combinations

import pytest

from mesh.edgeswap import EdgeSwap
from mesh.errors import Invalid
from mesh.factories import complete, cycle, path
from mesh.graph import Graph


def _random_graph(seed: int, n: int, p: float) -> Graph:
    rng = random.Random(seed)
    g = Graph()
    nodes = [str(i) for i in range(n)]
    for node in nodes:
        g.add_node(node)
    for a, b in combinations(nodes, 2):
        if rng.random() < p:
            g.add_edge(a, b, float(rng.randint(1, 5)))
    return g


class TestPreservation:
    def test_degrees_edge_count_and_simplicity_survive_many_swaps(self):
        g = _random_graph(617, 14, 0.3)
        es = EdgeSwap(g, seed=1)
        out = es.shuffle(400)
        assert es.degrees_preserved(out)
        assert out.edge_count() == g.edge_count()
        assert EdgeSwap.is_simple(out)
        assert out.node_count() == g.node_count()

    def test_the_input_graph_is_left_untouched(self):
        g = cycle(6)
        before = sorted((u, v) for u, v, _w in g.edges())
        EdgeSwap(g, seed=2).shuffle(50)
        assert sorted((u, v) for u, v, _w in g.edges()) == before

    def test_the_weight_travels_with_the_edge_it_replaced(self):
        g = _random_graph(619, 10, 0.4)
        out = EdgeSwap(g, seed=3).shuffle(100)
        assert sorted(w for _u, _v, w in out.edges()) == sorted(w for _u, _v, w in g.edges())


class TestMovement:
    def test_a_sparse_graph_ends_up_mostly_rewired(self):
        g = _random_graph(631, 20, 0.15)
        es = EdgeSwap(g, seed=4)
        out = es.shuffle(500)
        assert es.edges_changed(out) > g.edge_count() // 2
        assert es.accepted > 0

    def test_a_complete_graph_refuses_every_swap(self):
        g = complete(6)
        es = EdgeSwap(g, seed=5)
        out = es.shuffle(100)
        assert es.accepted == 0
        assert es.edges_changed(out) == 0

    def test_the_same_seed_gives_the_same_result(self):
        g = _random_graph(641, 12, 0.3)
        a = EdgeSwap(g, seed=7).shuffle(200)
        b = EdgeSwap(g, seed=7).shuffle(200)
        assert sorted(a.edges()) == sorted(b.edges())


class TestEdgeCases:
    def test_fewer_than_two_edges_returns_the_same_shape(self):
        es = EdgeSwap(path(2), seed=0)
        out = es.shuffle(10)
        assert out.edge_count() == 1
        assert es.attempted == 0

    def test_refusals(self):
        with pytest.raises(Invalid):
            EdgeSwap(Graph(directed=True))
        with pytest.raises(Invalid):
            EdgeSwap(cycle(4)).shuffle(-1)


class TestReport:
    def test_the_note_gives_the_acceptance_rate(self):
        es = EdgeSwap(complete(5), seed=0)
        es.shuffle(20)
        assert "0 of 20 swap(s) accepted (0%)" in es.note()
