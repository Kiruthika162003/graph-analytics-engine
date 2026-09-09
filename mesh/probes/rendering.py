"""Rendering probes: pictures and text held to the counts of the graph they draw.

An SVG parses and holds one circle per node and one line per edge, a
text grid is symmetric exactly for an undirected graph, a drawn tree
has one line per node, and a contract with no rules never breaches.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

from mesh.factories import cycle, star
from mesh.graph import Graph
from mesh.graphtext import TextRender
from mesh.graphvalidate import Contract
from mesh.layout import Layout
from mesh.probes.probe import Probe
from mesh.probes.registry import register
from mesh.svgrender import SvgRenderer

NS = "{http://www.w3.org/2000/svg}"


@register
def svg_holds_one_circle_per_node_and_one_line_per_edge() -> Probe:
    # the drawing must parse, and its element counts must be the graph's counts
    g = cycle(7)
    root = ET.fromstring(SvgRenderer(g, Layout(g).circle()).render())
    circles = len(root.findall(f"{NS}circle"))
    lines = len(root.findall(f"{NS}line"))
    return Probe(
        prober="svgrender",
        guarantee="a rendered SVG parses with one circle per node and one line per edge",
        holds=circles == g.node_count() and lines == g.edge_count(),
        readings={"circles": circles, "lines": lines},
    )


@register
def text_grid_symmetry_follows_direction() -> Probe:
    # an undirected grid is symmetric; a one-way arc breaks the symmetry
    undirected = TextRender(cycle(4)).grid_is_symmetric()
    g = Graph(directed=True)
    for n in "ab":
        g.add_node(n)
    g.add_edge("a", "b")
    directed = TextRender(g).grid_is_symmetric()
    return Probe(
        prober="graphtext",
        guarantee="the adjacency grid is symmetric for an undirected graph and not for an arc",
        holds=undirected and not directed,
        readings={"undirected": undirected, "directed": directed},
    )


@register
def drawn_tree_has_one_line_per_node() -> Probe:
    # every node of a star appears on exactly one line of the drawing
    g = star(6)
    text = TextRender(g).tree("0")
    return Probe(
        prober="graphtext",
        guarantee="a tree drawn with branch characters has exactly one line per node",
        holds=len(text.splitlines()) == g.node_count(),
        readings={"lines": len(text.splitlines()), "nodes": g.node_count()},
    )


@register
def empty_contract_never_breaches() -> Probe:
    # a contract with no requirements passes anything, including the empty graph
    breaches = Contract().check(Graph()) + Contract().check(cycle(3))
    return Probe(
        prober="graphvalidate",
        guarantee="a contract with no requirements reports no breach on any graph",
        holds=breaches == [],
        readings={"breaches": len(breaches)},
    )
