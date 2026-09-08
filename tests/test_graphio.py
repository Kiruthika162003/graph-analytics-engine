from __future__ import annotations

import pytest

from mesh.errors import Invalid
from mesh.factories import complete, cycle, grid, petersen, star
from mesh.graph import Graph
from mesh.graphio import (
    from_json,
    note,
    read_edge_list,
    same_graph,
    to_dot,
    to_json,
    write_edge_list,
)


def _weighted_digraph() -> Graph:
    g = Graph(directed=True)
    for n in "abc":
        g.add_node(n)
    g.add_edge("a", "b", 2.5)
    g.add_edge("b", "c", 1)
    g.add_node("lonely")
    return g


class TestEdgeList:
    def test_writing_then_reading_recovers_the_graph(self):
        g = _weighted_digraph()
        assert same_graph(read_edge_list(write_edge_list(g)), g)

    def test_an_isolated_node_survives_the_round_trip(self):
        back = read_edge_list(write_edge_list(_weighted_digraph()))
        assert back.has_node("lonely")
        assert back.degree("lonely") == 0

    def test_comments_and_blank_lines_are_ignored(self):
        text = "# a note\nundirected\n\na b 3\n# trailing\n"
        g = read_edge_list(text)
        assert g.weight("a", "b") == 3
        assert not g.directed

    def test_a_missing_weight_defaults_to_one(self):
        g = read_edge_list("undirected\nx y\n")
        assert g.weight("x", "y") == 1.0

    def test_every_factory_shape_round_trips(self):
        for g in (complete(5), cycle(6), star(4), grid(3), petersen()):
            assert same_graph(read_edge_list(write_edge_list(g)), g)

    def test_the_same_structure_writes_the_same_text(self):
        a = cycle(4)
        b = Graph()
        for n in ["3", "1", "2", "0"]:
            b.add_node(n)
        for u, v in [("2", "3"), ("0", "1"), ("3", "0"), ("1", "2")]:
            b.add_edge(u, v)
        assert write_edge_list(a) == write_edge_list(b)


class TestRefusals:
    def test_a_missing_header_is_refused(self):
        with pytest.raises(Invalid):
            read_edge_list("a b\n")

    def test_a_bad_weight_names_its_line(self):
        with pytest.raises(Invalid) as caught:
            read_edge_list("undirected\na b heavy\n")
        assert "line 2" in str(caught.value)

    def test_too_many_fields_names_its_line(self):
        with pytest.raises(Invalid) as caught:
            read_edge_list("directed\na b 1 extra\n")
        assert "line 2" in str(caught.value)

    def test_json_without_the_required_keys_is_refused(self):
        with pytest.raises(Invalid):
            from_json('{"nodes": []}')


class TestJson:
    def test_json_round_trips_with_weights_and_direction(self):
        g = _weighted_digraph()
        assert same_graph(from_json(to_json(g)), g)


class TestDot:
    def test_an_undirected_graph_uses_graph_and_double_dashes(self):
        dot = to_dot(cycle(3), name="tri")
        assert dot.startswith("graph tri {")
        assert '"0" -- "1";' in dot

    def test_a_directed_graph_uses_digraph_arrows_and_weight_labels(self):
        dot = to_dot(_weighted_digraph())
        assert dot.startswith("digraph mesh {")
        assert '"a" -> "b" [label="2.5"];' in dot
        assert '"b" -> "c";' in dot


class TestReport:
    def test_the_note_counts_lines_against_edges(self):
        # header, the lone node, and two edges: four lines, not the five I
        # first wrote down
        text = write_edge_list(_weighted_digraph())
        assert "4 non-blank line(s) read, 2 edge(s) found" in note(text, _weighted_digraph())
