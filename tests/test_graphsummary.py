from __future__ import annotations

import pytest

from mesh.factories import complete, cycle, path, star
from mesh.graph import Graph
from mesh.graphsummary import GraphSummary


class TestReadings:
    def test_density_and_degrees_of_a_complete_graph(self):
        s = GraphSummary(complete(5))
        assert s.density() == 1.0
        assert s.degrees() == (4, 4.0, 4)
        assert s.pieces() == 1

    def test_transitivity_is_one_on_a_clique_and_zero_on_a_tree(self):
        assert GraphSummary(complete(4)).transitivity() == pytest.approx(1.0)
        assert GraphSummary(star(4)).transitivity() == 0.0
        assert GraphSummary(path(5)).transitivity() == 0.0

    def test_pieces_count_components(self):
        g = Graph()
        for n in "abcde":
            g.add_node(n)
        g.add_edge("a", "b")
        g.add_edge("c", "d")
        assert GraphSummary(g).pieces() == 3


class TestClasses:
    def test_a_star_is_a_tree_and_every_recognised_class(self):
        found = GraphSummary(star(4)).classes()
        assert found == ["tree", "chordal", "split", "cograph", "threshold"]

    def test_a_long_cycle_is_in_no_class_and_a_square_is_a_cograph(self):
        assert GraphSummary(cycle(5)).classes() == []
        assert GraphSummary(cycle(4)).classes() == ["cograph"]

    def test_a_path_of_four_is_chordal_and_split_but_not_a_cograph(self):
        assert GraphSummary(path(4)).classes() == ["tree", "chordal", "split"]


class TestReport:
    def test_the_note_has_five_lines_for_a_connected_undirected_graph(self):
        lines = GraphSummary(cycle(6)).lines()
        assert len(lines) == 5
        assert lines[0] == "undirected graph with 6 node(s) and 6 edge(s), density 0.400"
        assert "radius 3, diameter 3" in lines[2]
        assert lines[3] == "0 triangle(s), transitivity 0.000"
        assert lines[4] == "classes: none recognised"

    def test_a_disconnected_graph_reports_absent_distances(self):
        g = Graph()
        for n in "abcd":
            g.add_node(n)
        g.add_edge("a", "b")
        assert "not connected" in GraphSummary(g).note()

    def test_a_directed_graph_gets_the_short_report(self):
        g = Graph(directed=True)
        g.add_node("x")
        g.add_node("y")
        g.add_edge("x", "y")
        lines = GraphSummary(g).lines()
        assert len(lines) == 3
        assert lines[0].startswith("directed graph with 2 node(s) and 1 edge(s), density 0.500")

    def test_the_empty_graph(self):
        assert GraphSummary(Graph()).note() == "empty graph: no nodes"
