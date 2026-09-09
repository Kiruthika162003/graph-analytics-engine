"""Change report: what happened to a graph's readings between two versions of it.

A graph that is tracked over time, a dependency graph week to week or
a friendship network term to term, is asked not what it looks like
but what changed. This module takes two versions and reports the
changes at three levels. The edge level lists nodes and edges added
and removed. The reading level compares the first-look numbers, edge
count, density, mean degree, pieces, triangles, and the largest
degree, and reports each as before, after, and the difference. The
node level ranks the nodes whose degree changed most, which is where
a reader finds the new hub or the departed one. A summary line states
whether the graph grew, shrank, or held its size, and whether it
became more or less connected, read from the piece count. The report
is built from the engine's own modules and checked on constructed
changes: adding a chord to a cycle adds one edge and one triangle
when it closes a three-cycle, removing a bridge adds a piece, and a
version compared with itself reports nothing changed at every level.
A directed version compared with an undirected one is refused, since
the readings do not line up.
"""

from __future__ import annotations

from mesh.errors import Invalid
from mesh.graph import Graph
from mesh.graphlets import Graphlets
from mesh.graphsummary import GraphSummary


class ChangeReport:
    def __init__(self, before: Graph, after: Graph) -> None:
        if before.directed != after.directed:
            raise Invalid("both versions must share their direction")
        self.before = before
        self.after = after
        old_nodes, new_nodes = set(before.nodes()), set(after.nodes())
        self.nodes_added = sorted(new_nodes - old_nodes)
        self.nodes_removed = sorted(old_nodes - new_nodes)
        old_edges = {self._key(u, v) for u, v, _w in before.edges()}
        new_edges = {self._key(u, v) for u, v, _w in after.edges()}
        self.edges_added = sorted(new_edges - old_edges)
        self.edges_removed = sorted(old_edges - new_edges)

    def _key(self, u: str, v: str) -> tuple[str, str]:
        return (u, v) if self.before.directed else (min(u, v), max(u, v))

    def readings(self) -> dict[str, tuple[float, float]]:
        out: dict[str, tuple[float, float]] = {}
        for label, fn in (
            ("edges", lambda g: float(g.edge_count())),
            ("density", lambda g: GraphSummary(g).density()),
            ("mean degree", lambda g: GraphSummary(g).degrees()[1]),
            ("largest degree", lambda g: float(GraphSummary(g).degrees()[2])),
            ("pieces", lambda g: float(GraphSummary(g).pieces())),
        ):
            out[label] = (fn(self.before), fn(self.after))
        if not self.before.directed:
            out["triangles"] = (
                float(Graphlets(self.before).triangles),
                float(Graphlets(self.after).triangles),
            )
        return out

    def degree_shifts(self, top: int = 3) -> list[tuple[str, int]]:
        shifts = []
        for n in set(self.before.nodes()) | set(self.after.nodes()):
            old = self.before.degree(n) if n in self.before.nodes() else 0
            new = self.after.degree(n) if n in self.after.nodes() else 0
            if old != new:
                shifts.append((n, new - old))
        shifts.sort(key=lambda p: (-abs(p[1]), p[0]))
        return shifts[:top]

    def unchanged(self) -> bool:
        return not (
            self.nodes_added or self.nodes_removed or self.edges_added or self.edges_removed
        )

    def lines(self) -> list[str]:
        if self.unchanged():
            return ["nothing changed at any level"]
        out = [
            f"nodes +{len(self.nodes_added)} -{len(self.nodes_removed)}, "
            f"edges +{len(self.edges_added)} -{len(self.edges_removed)}"
        ]
        for label, (old, new) in self.readings().items():
            out.append(f"{label}: {old:g} to {new:g} ({new - old:+g})")
        shifts = ", ".join(f"{n} {d:+d}" for n, d in self.degree_shifts())
        out.append(f"largest degree shifts: {shifts}" if shifts else "no degree shifted")
        grew = self.after.edge_count() - self.before.edge_count()
        size = "grew" if grew > 0 else "shrank" if grew < 0 else "held its size"
        pieces = self.readings()["pieces"]
        joined = (
            "became more connected"
            if pieces[1] < pieces[0]
            else "became less connected"
            if pieces[1] > pieces[0]
            else "kept its pieces"
        )
        out.append(f"the graph {size} and {joined}")
        return out

    def note(self) -> str:
        return "\n".join(self.lines())
