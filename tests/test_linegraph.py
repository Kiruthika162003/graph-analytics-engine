from __future__ import annotations

import random
from itertools import combinations

import pytest

from mesh.chromatic import ChromaticNumber
from mesh.edgecoloring import EdgeColoring
from mesh.errors import Invalid, Missing
from mesh.graph import Graph
from mesh.linegraph import LineGraph


def _star(leaves: int) -> Graph:
    g = Graph()
    g.add_node("hub")
    for i in range(leaves):
        g.add_node(f"l{i}")
        g.add_edge("hub", f"l{i}")
    return g


def _path(k: int) -> Graph:
    g = Graph()
    nodes = [str(i) for i in range(k)]
    for n in nodes:
        g.add_node(n)
    for i in range(k - 1):
        g.add_edge(nodes[i], nodes[i + 1])
    return g


class TestShape:
    def test_a_star_becomes_a_complete_graph(self):
        lg = LineGraph(_star(4)).line
        assert lg.node_count() == 4
        assert lg.edge_count() == 6

    def test_a_path_becomes_a_shorter_path(self):
        lg = LineGraph(_path(5)).line
        assert lg.node_count() == 4
        assert lg.edge_count() == 3

    def test_a_triangle_is_its_own_line_graph(self):
        g = Graph()
        for n in "abc":
            g.add_node(n)
        for a, b in combinations("abc", 2):
            g.add_edge(a, b)
        lg = LineGraph(g).line
        assert lg.node_count() == 3 and lg.edge_count() == 3

    def test_edge_nodes_are_named_by_sorted_endpoints(self):
        assert LineGraph(_path(3)).node_for("1", "0") == "0-1"

    def test_a_missing_edge_is_refused(self):
        with pytest.raises(Missing):
            LineGraph(_path(3)).node_for("0", "2")


class TestIdentities:
    def test_degree_and_edge_count_identities_hold_on_random_graphs(self):
        rng = random.Random(331)
        for _ in range(25):
            g = Graph()
            nodes = [str(i) for i in range(8)]
            for n in nodes:
                g.add_node(n)
            for a, b in combinations(nodes, 2):
                if rng.random() < 0.4:
                    g.add_edge(a, b)
            lg = LineGraph(g)
            assert lg.degree_identity_holds()
            assert lg.edge_count_identity_holds()

    def test_chromatic_number_of_the_line_graph_is_the_edge_chromatic_index(self):
        rng = random.Random(337)
        for _ in range(10):
            g = Graph()
            left = [f"l{i}" for i in range(3)]
            right = [f"r{i}" for i in range(3)]
            for n in left + right:
                g.add_node(n)
            for u in left:
                for v in right:
                    if rng.random() < 0.7:
                        g.add_edge(u, v)
            if g.edge_count() == 0:
                continue
            # bipartite, so the exact edge coloring uses the max degree
            by_line_graph = ChromaticNumber(LineGraph(g).line).value
            assert by_line_graph == EdgeColoring(g).color_count()


class TestRefusal:
    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            LineGraph(Graph(directed=True))


class TestReport:
    def test_the_note_states_both_sizes(self):
        note = LineGraph(_star(4)).note()
        assert "line graph of 4 node(s) and 6 edge(s) from 5 and 4" in note
