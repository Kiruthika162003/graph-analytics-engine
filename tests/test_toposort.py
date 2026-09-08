from __future__ import annotations

import pytest

from mesh.errors import Cyclic, Invalid
from mesh.graph import Graph
from mesh.toposort import TopologicalSort


def _dag() -> Graph:
    # shirt -> tie -> jacket, shirt -> belt, pants -> belt, pants -> jacket
    g = Graph(directed=True)
    for n in ["shirt", "tie", "jacket", "belt", "pants"]:
        g.add_node(n)
    for u, v in [("shirt", "tie"), ("tie", "jacket"), ("shirt", "belt"),
                 ("pants", "belt"), ("pants", "jacket")]:
        g.add_edge(u, v)
    return g


class TestOrder:
    def test_every_edge_points_forward_in_the_order(self):
        g = _dag()
        order = TopologicalSort(g).order()
        rank = {n: i for i, n in enumerate(order)}
        for u, v, _w in g.edges():
            assert rank[u] < rank[v]

    def test_the_order_contains_every_node_once(self):
        g = _dag()
        order = TopologicalSort(g).order()
        assert sorted(order) == sorted(g.nodes())

    def test_the_order_is_deterministic(self):
        g = _dag()
        assert TopologicalSort(g).order() == TopologicalSort(g).order()


class TestCycle:
    def test_a_cyclic_graph_is_refused(self):
        g = Graph(directed=True)
        for n in "abc":
            g.add_node(n)
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        g.add_edge("c", "a")
        with pytest.raises(Cyclic) as caught:
            TopologicalSort(g).order()
        assert "cycle" in str(caught.value)

    def test_is_acyclic_is_true_for_a_dag(self):
        assert TopologicalSort(_dag()).is_acyclic()

    def test_is_acyclic_is_false_for_a_cycle(self):
        g = Graph(directed=True)
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b")
        g.add_edge("b", "a")
        assert not TopologicalSort(g).is_acyclic()

    def test_only_the_stuck_nodes_are_named(self):
        # a clean chain feeding a separate cycle: only the cycle is stuck
        g = Graph(directed=True)
        for n in "xyabc":
            g.add_node(n)
        g.add_edge("x", "y")  # a clean edge, both should sort out
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        g.add_edge("c", "a")  # a, b, c form a cycle
        with pytest.raises(Cyclic) as caught:
            TopologicalSort(g).order()
        message = str(caught.value)
        assert "a" in message and "x" not in message.split("[")[1]


class TestConfig:
    def test_an_undirected_graph_is_refused(self):
        g = Graph(directed=False)
        with pytest.raises(Invalid):
            TopologicalSort(g)

    def test_the_note_counts_sources(self):
        # shirt and pants both have in-degree zero
        assert "2 source node(s)" in TopologicalSort(_dag()).note()
