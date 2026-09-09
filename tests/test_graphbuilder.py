from __future__ import annotations

import pytest

from mesh.errors import Invalid
from mesh.factories import path
from mesh.graphbuilder import GraphBuilder
from mesh.graphio import same_graph


def _by_calls():
    return GraphBuilder().nodes(["a", "b", "c"]).edge("a", "b").edge("b", "c", 2.0).build()


class TestFourWays:
    def test_calls_text_pairs_and_a_shape_all_give_the_same_graph(self):
        by_calls = _by_calls()
        by_text = GraphBuilder().text("a b\nb c 2\n").build()
        by_pairs = GraphBuilder().nodes("abc").edges([("a", "b"), ("b", "c", 2.0)]).build()
        assert same_graph(by_calls, by_text)
        assert same_graph(by_calls, by_pairs)
        assert by_calls.weight("b", "c") == 2.0

    def test_a_shape_joins_with_a_prefix_and_can_be_wired_to_the_rest(self):
        g = GraphBuilder().node("hub").shape("cycle", 4, prefix="r").edge("hub", "r0").build()
        assert g.node_count() == 5
        assert g.edge_count() == 5
        assert g.has_edge("hub", "r0")

    def test_text_ignores_comments_and_blank_lines(self):
        g = GraphBuilder().text("# office\n\nada ben\n\nlonely\n").build()
        assert g.node_count() == 3
        assert g.edge_count() == 1


class TestRefusals:
    def test_duplicate_and_bad_names_are_refused(self):
        with pytest.raises(Invalid, match="already added"):
            GraphBuilder().node("a").node("a")
        with pytest.raises(Invalid):
            GraphBuilder().node(" a")
        with pytest.raises(Invalid):
            GraphBuilder().node("")

    def test_edges_need_declared_nodes_and_numeric_weights(self):
        with pytest.raises(Invalid, match="never added"):
            GraphBuilder().node("a").edge("a", "b")
        with pytest.raises(Invalid, match="not a number"):
            GraphBuilder().nodes("ab").edge("a", "b", "heavy")  # type: ignore[arg-type]
        with pytest.raises(Invalid, match="line 1: weight"):
            GraphBuilder().text("a b heavy")

    def test_unknown_shapes_and_wrong_directions_are_refused(self):
        with pytest.raises(Invalid, match="no shape"):
            GraphBuilder().shape("dodecahedron")
        with pytest.raises(Invalid, match="wrong direction"):
            GraphBuilder(directed=True).shape("path", 3)

    def test_a_builder_builds_once(self):
        b = GraphBuilder().node("a")
        b.build()
        with pytest.raises(Invalid):
            b.build()
        with pytest.raises(Invalid):
            b.node("b")

    def test_a_malformed_line_names_its_number(self):
        with pytest.raises(Invalid, match="line 2"):
            GraphBuilder().text("a b\na b c d")


class TestReport:
    def test_the_note_counts_steps(self):
        b = GraphBuilder().nodes("abc").edge("a", "b")
        assert b.note() == "undirected graph of 3 node(s) and 1 edge(s) from 4 step(s)"
        assert same_graph(b.build(), GraphBuilder().text("a b\nc").build())
        assert same_graph(GraphBuilder().shape("path", 3).build(), path(3))
