from __future__ import annotations

import pytest

from mesh.dfs import DFS
from mesh.errors import Missing
from mesh.graph import Graph


def _dag() -> Graph:
    # a -> b -> d, a -> c -> d
    g = Graph(directed=True)
    for n in "abcd":
        g.add_node(n)
    for u, v in [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")]:
        g.add_edge(u, v)
    return g


class TestTimestamps:
    def test_every_node_is_discovered_before_it_is_finished(self):
        d = DFS(_dag(), "a")
        for node in d.reachable_nodes():
            assert d.discover[node] < d.finish[node]

    def test_the_source_is_discovered_first(self):
        d = DFS(_dag(), "a")
        assert d.preorder[0] == "a"

    def test_the_parenthesis_theorem_marks_descendants(self):
        d = DFS(_dag(), "a")
        # d is reachable below a, so a's interval contains d's
        assert d.is_ancestor("a", "d")

    def test_unrelated_nodes_are_not_ancestors(self):
        d = DFS(_dag(), "a")
        # b and c are siblings; neither contains the other
        assert not d.is_ancestor("b", "c")
        assert not d.is_ancestor("c", "b")


class TestBackEdge:
    def test_a_dag_has_no_back_edge(self):
        assert not DFS(_dag(), "a").has_back_edge

    def test_a_directed_cycle_has_a_back_edge(self):
        g = Graph(directed=True)
        for n in "abc":
            g.add_node(n)
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        g.add_edge("c", "a")  # closes the cycle
        assert DFS(g, "a").has_back_edge

    def test_an_undirected_triangle_has_a_back_edge(self):
        g = Graph()
        for n in "abc":
            g.add_node(n)
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        g.add_edge("c", "a")
        assert DFS(g, "a").has_back_edge

    def test_an_undirected_tree_has_no_back_edge(self):
        # a path is a tree; the parent edge must not be mistaken for a cycle
        g = Graph()
        for n in "abc":
            g.add_node(n)
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        assert not DFS(g, "a").has_back_edge


class TestTopoOrder:
    def test_reversed_postorder_respects_every_edge(self):
        g = _dag()
        d = DFS(g, "a")
        order = list(reversed(d.postorder))
        rank = {node: i for i, node in enumerate(order)}
        for u, v, _w in g.edges():
            assert rank[u] < rank[v]


class TestRefusal:
    def test_a_missing_source_is_refused(self):
        with pytest.raises(Missing):
            DFS(_dag(), "ghost")

    def test_the_note_reports_the_back_edge_flag(self):
        assert "back edge seen: False" in DFS(_dag(), "a").note()
