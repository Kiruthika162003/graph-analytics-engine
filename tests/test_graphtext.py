from __future__ import annotations

import pytest

from mesh.errors import Invalid
from mesh.factories import cycle, path, star
from mesh.graph import Graph
from mesh.graphtext import TextRender


class TestGrid:
    def test_a_triangle_prints_a_symmetric_grid_with_an_empty_diagonal(self):
        text = TextRender(cycle(3)).grid()
        assert text.splitlines() == [
            "  0 1 2",
            "0 . # #",
            "1 # . #",
            "2 # # .",
        ]
        assert TextRender(cycle(3)).grid_is_symmetric()

    def test_a_directed_grid_is_not_symmetric(self):
        g = Graph(directed=True)
        for n in "ab":
            g.add_node(n)
        g.add_edge("a", "b")
        tr = TextRender(g)
        assert not tr.grid_is_symmetric()
        assert tr.grid().splitlines()[1] == "a . #"

    def test_long_names_are_cut_with_a_mark(self):
        g = Graph()
        g.add_node("averylongname")
        g.add_node("b")
        g.add_edge("averylongname", "b")
        lines = TextRender(g).grid().splitlines()
        assert lines[1].startswith("averylo~")


class TestHistogram:
    def test_bars_scale_to_the_largest_count(self):
        text = TextRender(star(4)).histogram(width=8)
        assert text.splitlines() == [
            "  1 | ######## 4",
            "  4 | ## 1",
        ]

    def test_a_regular_graph_has_one_bar_and_an_empty_graph_none(self):
        assert TextRender(cycle(5)).histogram().splitlines() == ["  2 | #################### 5"]
        assert TextRender(Graph()).histogram() == "no nodes"


class TestTree:
    def test_a_small_tree_is_drawn_with_branches(self):
        g = Graph()
        for n in ("root", "a", "b", "a1", "a2"):
            g.add_node(n)
        for u, v in [("root", "a"), ("root", "b"), ("a", "a1"), ("a", "a2")]:
            g.add_edge(u, v)
        assert TextRender(g).tree("root").splitlines() == [
            "root",
            "|-- a",
            "|   |-- a1",
            "|   `-- a2",
            "`-- b",
        ]

    def test_the_line_count_is_the_node_count_and_a_path_rooted_mid_way(self):
        text = TextRender(path(5)).tree("2")
        assert len(text.splitlines()) == 5
        assert text.splitlines()[1] == "|-- 1"

    def test_non_trees_and_unknown_roots_are_refused(self):
        with pytest.raises(Invalid):
            TextRender(cycle(4)).tree("0")
        with pytest.raises(Invalid):
            TextRender(path(3)).tree("zz")
        # the guess was that two separate edges plus a pendant would read as not
        # connected; five nodes on three edges fail the edge count first. a triangle
        # with an isolated node has n minus one edges and is the disconnected case
        g = cycle(3)
        g.add_node("d")
        with pytest.raises(Invalid, match="not connected"):
            TextRender(g).tree("0")


class TestReport:
    def test_the_note_counts_lines_and_bars(self):
        assert TextRender(star(3)).note() == (
            "text renderings of 4 node(s): grid 5 line(s), histogram 2 bar(s)"
        )
