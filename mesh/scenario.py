"""Scenarios: what happens to the readings if this node or edge were gone, or this edge added.

The question behind most graph analysis is a counterfactual: what
if this server failed, what if this road closed, what if these two
people were introduced. A scenario applies one such change to a copy
of the graph, runs the same readings on both, and reports the
deltas, so the answer is a table of before and after rather than a
new graph the reader must inspect. The readings are the cheap ones a
what-if usually wants: the number of pieces, the largest piece, the
diameter of the largest piece, the mean distance within it, and the
edge count. Three kinds of change are supported, node removal, edge
removal, and edge addition, and each is refused when it names
something absent or, for addition, something already present. A
sweep runs one kind of change over every candidate and ranks them by
the reading that changed most, which is how a reader finds the most
damaging failure or the most helpful new link without guessing. The
tests hold the facts a counterfactual must respect: removing a
bridge raises the piece count by one, adding a chord to a long path
lowers the diameter, removing a leaf changes nothing but the counts,
the sweep over a star ranks the hub first, and a scenario applied
and reversed leaves the original untouched.
"""

from __future__ import annotations

from collections import deque

from mesh.errors import Invalid
from mesh.graph import Graph


def _copy_without(graph: Graph, node: str | None, edge: tuple[str, str] | None) -> Graph:
    g = Graph(directed=graph.directed)
    for n in graph.nodes():
        if n != node:
            g.add_node(n)
    for u, v, w in graph.edges():
        if node in (u, v):
            continue
        if edge is not None and ({u, v} == set(edge) if not graph.directed else (u, v) == edge):
            continue
        g.add_edge(u, v, w)
    return g


def readings(graph: Graph) -> dict[str, float]:
    seen: set[str] = set()
    pieces = 0
    largest: list[str] = []
    for start in graph.nodes():
        if start in seen:
            continue
        pieces += 1
        piece = [start]
        seen.add(start)
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for other in graph.neighbors(node):
                if other not in seen:
                    seen.add(other)
                    piece.append(other)
                    queue.append(other)
        if len(piece) > len(largest):
            largest = piece
    diameter = 0.0
    total = 0.0
    pairs = 0
    inside = set(largest)
    for start in largest:
        dist = {start: 0}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for other in graph.neighbors(node):
                if other in inside and other not in dist:
                    dist[other] = dist[node] + 1
                    queue.append(other)
        for other, d in dist.items():
            if other != start:
                diameter = max(diameter, d)
                total += d
                pairs += 1
    return {
        "pieces": float(pieces),
        "largest": float(len(largest)),
        "diameter": diameter if largest else 0.0,
        "mean distance": total / pairs if pairs else 0.0,
        "edges": float(graph.edge_count()),
    }


class Scenario:
    def __init__(self, graph: Graph) -> None:
        self.graph = graph
        self.before = readings(graph)

    def remove_node(self, node: str) -> dict[str, tuple[float, float]]:
        if node not in self.graph.nodes():
            raise Invalid(f"no node '{node}' to remove")
        return self._compare(_copy_without(self.graph, node, None))

    def remove_edge(self, u: str, v: str) -> dict[str, tuple[float, float]]:
        if not self.graph.has_edge(u, v):
            raise Invalid(f"no edge {u}-{v} to remove")
        return self._compare(_copy_without(self.graph, None, (u, v)))

    def add_edge(self, u: str, v: str, weight: float = 1.0) -> dict[str, tuple[float, float]]:
        for n in (u, v):
            if n not in self.graph.nodes():
                raise Invalid(f"no node '{n}' to join")
        if self.graph.has_edge(u, v):
            raise Invalid(f"edge {u}-{v} is already present")
        g = _copy_without(self.graph, None, None)
        g.add_edge(u, v, weight)
        return self._compare(g)

    def _compare(self, after: Graph) -> dict[str, tuple[float, float]]:
        later = readings(after)
        return {k: (self.before[k], later[k]) for k in self.before}

    def sweep_nodes(self, by: str = "largest") -> list[tuple[str, float]]:
        if by not in self.before:
            raise Invalid(f"no reading called '{by}'")
        ranked = []
        for node in self.graph.nodes():
            before, after = self.remove_node(node)[by]
            ranked.append((node, after - before))
        return sorted(ranked, key=lambda p: (-abs(p[1]), p[0]))

    def sweep_additions(self, by: str = "diameter") -> list[tuple[tuple[str, str], float]]:
        if by not in self.before:
            raise Invalid(f"no reading called '{by}'")
        ranked = []
        nodes = self.graph.nodes()
        for i, a in enumerate(nodes):
            for b in nodes[i + 1 :]:
                if self.graph.has_edge(a, b):
                    continue
                before, after = self.add_edge(a, b)[by]
                ranked.append(((a, b), after - before))
        return sorted(ranked, key=lambda p: (p[1], p[0]))

    def untouched(self) -> bool:
        return readings(self.graph) == self.before

    def note(self, table: dict[str, tuple[float, float]]) -> str:
        parts = [f"{k} {a:g} to {b:g}" for k, (a, b) in table.items() if a != b]
        return "; ".join(parts) if parts else "nothing changed"
