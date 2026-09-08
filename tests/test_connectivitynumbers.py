from __future__ import annotations

import random
from itertools import combinations

import pytest

from mesh.connectivitynumbers import ConnectivityNumbers
from mesh.errors import Invalid
from mesh.graph import Graph


def _cycle(k: int) -> Graph:
    g = Graph()
    nodes = [str(i) for i in range(k)]
    for n in nodes:
        g.add_node(n)
    for i in range(k):
        g.add_edge(nodes[i], nodes[(i + 1) % k])
    return g


def _complete(k: int) -> Graph:
    g = Graph()
    nodes = [str(i) for i in range(k)]
    for n in nodes:
        g.add_node(n)
    for a, b in combinations(nodes, 2):
        g.add_edge(a, b)
    return g


class TestNumbers:
    def test_a_ring_has_both_numbers_at_two(self):
        cn = ConnectivityNumbers(_cycle(6))
        assert cn.edge_connectivity == 2
        assert cn.vertex_connectivity == 2

    def test_a_tree_has_both_numbers_at_one(self):
        g = Graph()
        for n in "abcd":
            g.add_node(n)
        for u, v in [("a", "b"), ("b", "c"), ("b", "d")]:
            g.add_edge(u, v)
        cn = ConnectivityNumbers(g)
        assert (cn.edge_connectivity, cn.vertex_connectivity) == (1, 1)

    def test_a_complete_graph_is_n_minus_one_connected(self):
        cn = ConnectivityNumbers(_complete(5))
        assert cn.edge_connectivity == 4
        assert cn.vertex_connectivity == 4
        assert cn.is_k_connected(4)

    def test_a_disconnected_graph_has_zero(self):
        g = _cycle(3)
        g.add_node("island")
        cn = ConnectivityNumbers(g)
        assert (cn.edge_connectivity, cn.vertex_connectivity) == (0, 0)

    def test_redundant_edges_through_one_hub_show_a_gap(self):
        # two triangles joined only at a shared hub: edges are doubly
        # redundant around each triangle, but the hub alone splits them
        g = Graph()
        for n in "abhcd":
            g.add_node(n)
        for u, v in [("a", "b"), ("b", "h"), ("h", "a"), ("h", "c"), ("c", "d"), ("d", "h")]:
            g.add_edge(u, v)
        cn = ConnectivityNumbers(g)
        assert cn.vertex_connectivity == 1
        assert cn.edge_connectivity == 2
        assert "gap of 1" in cn.note()


class TestWhitney:
    def test_the_inequality_chain_holds_on_random_graphs(self):
        rng = random.Random(313)
        for _ in range(25):
            g = Graph()
            nodes = [str(i) for i in range(7)]
            for n in nodes:
                g.add_node(n)
            for a, b in combinations(nodes, 2):
                if rng.random() < 0.5:
                    g.add_edge(a, b)
            assert ConnectivityNumbers(g).whitney_holds()


class TestRefusals:
    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            ConnectivityNumbers(Graph(directed=True))

    def test_a_single_node_is_refused(self):
        g = Graph()
        g.add_node("a")
        with pytest.raises(Invalid):
            ConnectivityNumbers(g)


class TestReport:
    def test_the_note_states_the_chain(self):
        note = ConnectivityNumbers(_cycle(5)).note()
        assert "vertex 2 <= edge 2 <= min degree 2" in note
