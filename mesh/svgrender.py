"""SVG rendering: a graph and its layout written as a drawing any browser opens.

A picture is the reading most people trust first, and SVG is text,
so the engine can produce one without a drawing library. The
renderer takes a graph and a layout of positions in the unit square,
scales them to a canvas with a margin, draws each edge as a line and
each node as a circle with its name beside it, and writes directed
edges with an arrowhead marker. Optional per-node colors let a
caller show communities, and optional per-edge widths let weights or
flows show as thickness. The output is a complete SVG document as a
string, which the tests parse with the standard library's XML module
to count the elements and check the attributes, since a drawing that
does not parse is not a drawing. The identities held are the ones a
picture must satisfy: one circle and one label per node, one line per
edge, every coordinate inside the canvas, an arrowhead marker present
exactly when the graph is directed, and a legend line listing the
colors used when any were given. A layout missing a node is refused
by name, and so is a width or color for an edge or node the graph
does not have.
"""

from __future__ import annotations

from mesh.errors import Invalid
from mesh.graph import Graph

Point = tuple[float, float]


class SvgRenderer:
    def __init__(self, graph: Graph, positions: dict[str, Point], size: int = 400) -> None:
        if size < 50:
            raise Invalid("a canvas smaller than fifty pixels cannot hold labels")
        missing = [n for n in graph.nodes() if n not in positions]
        if missing:
            raise Invalid(f"no position for '{missing[0]}'")
        self.graph = graph
        self.positions = positions
        self.size = size
        self.margin = 30
        self.colors: dict[str, str] = {}
        self.widths: dict[tuple[str, str], float] = {}

    def color(self, node: str, fill: str) -> SvgRenderer:
        if node not in self.graph.nodes():
            raise Invalid(f"'{node}' is not a node of the graph")
        self.colors[node] = fill
        return self

    def width(self, u: str, v: str, stroke: float) -> SvgRenderer:
        if not self.graph.has_edge(u, v):
            raise Invalid(f"no edge {u}-{v} to widen")
        if stroke <= 0:
            raise Invalid("a stroke width must be positive")
        self.widths[(u, v)] = stroke
        return self

    def _scale(self, point: Point) -> Point:
        span = self.size - 2 * self.margin
        return (self.margin + point[0] * span, self.margin + point[1] * span)

    def _edge_width(self, u: str, v: str) -> float:
        return self.widths.get((u, v), self.widths.get((v, u), 1.5))

    def render(self) -> str:
        lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.size}" height="{self.size}" '
            f'viewBox="0 0 {self.size} {self.size}">',
        ]
        if self.graph.directed:
            lines.append(
                '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" '
                'orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#444"/></marker></defs>'
            )
        for u, v, _w in self.graph.edges():
            (x1, y1), (x2, y2) = self._scale(self.positions[u]), self._scale(self.positions[v])
            marker = ' marker-end="url(#arrow)"' if self.graph.directed else ""
            lines.append(
                f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#444" '
                f'stroke-width="{self._edge_width(u, v):.1f}"{marker}/>'
            )
        for node in self.graph.nodes():
            x, y = self._scale(self.positions[node])
            fill = self.colors.get(node, "#dde")
            lines.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="{fill}" stroke="#333"/>'
            )
            lines.append(f'<text x="{x + 11:.1f}" y="{y + 4:.1f}" font-size="11">{node}</text>')
        if self.colors:
            used = sorted(set(self.colors.values()))
            lines.append(
                f'<text x="{self.margin}" y="{self.size - 8}" font-size="10">'
                f"colors: {', '.join(used)}</text>"
            )
        lines.append("</svg>")
        return "\n".join(lines)

    def note(self) -> str:
        size = f"{self.graph.node_count()} node(s) and {self.graph.edge_count()} edge(s)"
        return (
            f"svg of {size} on a {self.size} pixel canvas, {len(self.colors)} colored, "
            f"{len(self.widths)} widened"
        )
