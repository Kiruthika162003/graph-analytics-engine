from __future__ import annotations

from itertools import combinations

import pytest

from mesh.errors import MeshError
from mesh.factories import cycle, star
from mesh.graph import Graph
from mesh.graphreport import SECTIONS, FullReport


def _two_teams() -> Graph:
    g = Graph()
    for team in ("abcd", "wxyz"):
        for n in team:
            g.add_node(n)
        for a, b in combinations(team, 2):
            g.add_edge(a, b)
    g.add_edge("d", "w")
    return g


class TestFull:
    def test_a_small_undirected_graph_fills_every_section(self):
        report = FullReport(_two_teams())
        assert report.written == len(SECTIONS)
        assert report.skipped == 0
        headings = [line for line in report.lines if line.startswith("== ")]
        assert len(headings) == len(SECTIONS)
        assert report.note() == f"{len(SECTIONS)} section(s) written, 0 skipped"
        assert "  verdict: clear communities" in report.lines

    def test_the_last_line_counts_match_the_headings(self):
        report = FullReport(star(4))
        headings = sum(1 for line in report.lines if line.startswith("== "))
        assert headings == report.written + report.skipped

    def test_sections_can_be_asked_for_by_name(self):
        report = FullReport(cycle(5), names=["summary", "classes"])
        assert report.written == 2
        assert report.section("classes") == ["classes: none recognised"]
        with pytest.raises(MeshError):
            report.section("weather")
        with pytest.raises(MeshError):
            FullReport(cycle(5), names=["weather"])


class TestSkips:
    def test_a_directed_graph_skips_what_it_must_and_says_so(self):
        g = Graph(directed=True)
        for n in "abc":
            g.add_node(n)
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        report = FullReport(g)
        assert report.skipped > 0
        assert any(line.startswith("  skipped: ") for line in report.lines)
        assert report.written + report.skipped == len(SECTIONS)

    def test_the_empty_graph_reports_rather_than_raising(self):
        report = FullReport(Graph())
        assert report.written + report.skipped == len(SECTIONS)
        assert report.text().endswith("skipped")
        assert "== summary" in report.lines
        assert "  empty graph: no nodes" in report.lines
