"""Text rendering: an adjacency grid, a degree histogram, and a tree drawn with characters.

Not every terminal opens an SVG, and a graph small enough to read is
worth printing as characters. Three renderings live here. The
adjacency grid prints a row per node and a column per node with a
mark where an edge sits, which shows symmetry, density, and blocks
at a glance and is the picture a matrix-minded reader wants. The
degree histogram prints one bar per degree value, scaled to a fixed
width, which shows the tail. And the tree rendering draws a rooted
tree with indentation and branch characters, one node per line, the
way a file listing does, which is the natural view of a hierarchy
and refuses a graph that is not a tree or a root that is not in it.
All three return a string with newlines, so they print or compare as
text, and the tests check them line by line against hand-drawn
expectations for a triangle, a star, and a small tree, and check the
grid's symmetry on an undirected graph, the histogram's bar lengths
against the degree counts, and the tree's line count against the
node count. Names longer than the column width are cut with a mark,
so a wide name cannot break the grid.
"""

from __future__ import annotations

from collections import Counter

from mesh.errors import Invalid
from mesh.graph import Graph


class TextRender:
    def __init__(self, graph: Graph) -> None:
        self.graph = graph
        self.nodes = graph.nodes()

    @staticmethod
    def _fit(name: str, width: int) -> str:
        return name if len(name) <= width else name[: width - 1] + "~"

    def grid(self, mark: str = "#", blank: str = ".") -> str:
        width = max((len(n) for n in self.nodes), default=1)
        width = min(width, 8)
        heads = " ".join(self._fit(n, width).rjust(width) for n in self.nodes)
        lines = [" " * (width + 1) + heads]
        for a in self.nodes:
            cells = [
                (mark if self.graph.has_edge(a, b) else blank).rjust(width) for b in self.nodes
            ]
            lines.append(self._fit(a, width).ljust(width) + " " + " ".join(cells))
        return "\n".join(lines)

    def grid_is_symmetric(self) -> bool:
        rows = self.grid().splitlines()[1:]
        cells = [row.split()[1:] for row in rows]
        n = len(cells)
        return all(cells[i][j] == cells[j][i] for i in range(n) for j in range(n))

    def histogram(self, width: int = 20) -> str:
        counts = Counter(self.graph.degree(n) for n in self.nodes)
        if not counts:
            return "no nodes"
        top = max(counts.values())
        lines = []
        for degree in sorted(counts):
            bar = "#" * max(1, round(counts[degree] / top * width))
            lines.append(f"{degree:>3} | {bar} {counts[degree]}")
        return "\n".join(lines)

    def tree(self, root: str) -> str:
        if root not in self.nodes:
            raise Invalid(f"'{root}' is not a node of the graph")
        if self.graph.edge_count() != self.graph.node_count() - 1:
            raise Invalid("a tree has exactly n minus one edges")
        lines = [root]
        seen = {root}

        def walk(node: str, prefix: str) -> None:
            children = sorted(m for m in self.graph.neighbors(node) if m not in seen)
            for i, child in enumerate(children):
                seen.add(child)
                last = i == len(children) - 1
                lines.append(prefix + ("`-- " if last else "|-- ") + child)
                walk(child, prefix + ("    " if last else "|   "))

        walk(root, "")
        if len(seen) != len(self.nodes):
            raise Invalid("the graph is not connected, so it is not a tree")
        return "\n".join(lines)

    def note(self) -> str:
        bars = len(self.histogram().splitlines())
        return (
            f"text renderings of {len(self.nodes)} node(s): grid {len(self.nodes) + 1} "
            f"line(s), histogram {bars} bar(s)"
        )
