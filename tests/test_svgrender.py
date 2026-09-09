from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from mesh.errors import Invalid
from mesh.factories import cycle, path
from mesh.graph import Graph
from mesh.layout import Layout
from mesh.svgrender import SvgRenderer

NS = "{http://www.w3.org/2000/svg}"


def _parse(text: str) -> ET.Element:
    return ET.fromstring(text)


class TestElements:
    def test_one_circle_and_label_per_node_and_one_line_per_edge(self):
        g = cycle(5)
        root = _parse(SvgRenderer(g, Layout(g).circle()).render())
        assert len(root.findall(f"{NS}circle")) == 5
        assert len(root.findall(f"{NS}text")) == 5
        assert len(root.findall(f"{NS}line")) == 5
        assert root.find(f"{NS}defs") is None

    def test_every_coordinate_stays_inside_the_canvas(self):
        g = path(6)
        root = _parse(SvgRenderer(g, Layout(g).spring(rounds=50, seed=2), size=300).render())
        for circle in root.findall(f"{NS}circle"):
            assert 0 <= float(circle.get("cx")) <= 300
            assert 0 <= float(circle.get("cy")) <= 300

    def test_a_directed_graph_gets_an_arrow_marker_on_every_line(self):
        g = Graph(directed=True)
        for n in "abc":
            g.add_node(n)
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        root = _parse(SvgRenderer(g, Layout(g).layered()).render())
        assert root.find(f"{NS}defs") is not None
        lines = root.findall(f"{NS}line")
        assert all(line.get("marker-end") == "url(#arrow)" for line in lines)


class TestDecoration:
    def test_colors_and_widths_appear_with_a_legend(self):
        g = path(3)
        r = SvgRenderer(g, Layout(g).circle()).color("0", "#f00").color("2", "#0f0")
        r.width("0", "1", 4.0)
        root = _parse(r.render())
        fills = [c.get("fill") for c in root.findall(f"{NS}circle")]
        assert fills.count("#f00") == 1 and fills.count("#0f0") == 1
        widths = sorted(float(line.get("stroke-width")) for line in root.findall(f"{NS}line"))
        assert widths == [1.5, 4.0]
        legend = root.findall(f"{NS}text")[-1].text
        assert legend == "colors: #0f0, #f00"

    def test_a_width_given_in_the_other_orientation_still_applies(self):
        g = path(2)
        r = SvgRenderer(g, Layout(g).circle()).width("1", "0", 3.0)
        root = _parse(r.render())
        assert float(root.find(f"{NS}line").get("stroke-width")) == 3.0


class TestRefusal:
    def test_missing_positions_unknown_names_and_tiny_canvases_are_refused(self):
        g = path(3)
        with pytest.raises(Invalid, match="no position for '2'"):
            SvgRenderer(g, {"0": (0, 0), "1": (1, 1)})
        r = SvgRenderer(g, Layout(g).circle())
        with pytest.raises(Invalid):
            r.color("zz", "#000")
        with pytest.raises(Invalid):
            r.width("0", "2", 2.0)
        with pytest.raises(Invalid):
            r.width("0", "1", 0.0)
        with pytest.raises(Invalid):
            SvgRenderer(g, Layout(g).circle(), size=10)


class TestReport:
    def test_the_note_counts_the_decorations(self):
        g = cycle(4)
        r = SvgRenderer(g, Layout(g).circle()).color("0", "#abc").width("0", "1", 2.0)
        assert r.note() == (
            "svg of 4 node(s) and 4 edge(s) on a 400 pixel canvas, 1 colored, 1 widened"
        )
