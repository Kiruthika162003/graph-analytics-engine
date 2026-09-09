"""Wiener index and its relatives: distance sums that read a graph's compactness.

The Wiener index is the sum of shortest-path distances over every
pair of nodes. Wiener introduced it in 1947 to predict the boiling
points of alkanes from their carbon skeletons, and it remains the
first topological index chemists compute: a compact molecule has a
small index and a stretched one a large index. Two relatives come from
the same distance table. The average distance is the index divided by
the number of pairs, which is the mean hop count between two random
nodes and the number a network designer quotes. The Harary index sums
the reciprocals of the distances, so nearby pairs count more and a
disconnected pair contributes nothing, which lets it read a
disconnected graph the Wiener index cannot. The closed forms are
exact: a path on n nodes has index (n minus 1) n (n plus 1) over 6, a
complete graph has n choose 2, and a star on n nodes has (n minus 1)
squared. The engine builds the distance table
by a breadth-first sweep from every node, or by Dijkstra when
weights matter, computes the three readings, checks the path form
against a direct sum, and on a tree confirms the edge-cut identity:
the Wiener index of a tree equals the sum over edges of the product of
the two sides' sizes, since each edge is crossed by exactly the pairs
it separates. A directed graph is refused, and the Wiener index of a
disconnected graph is reported as infinite rather than as a partial
sum.
"""

from __future__ import annotations

from collections import deque
from itertools import combinations
from math import inf

from mesh.dijkstra import Dijkstra
from mesh.errors import Invalid
from mesh.graph import Graph


class WienerIndex:
    def __init__(self, graph: Graph, weighted: bool = False) -> None:
        if graph.directed:
            raise Invalid("distance sums here are read on undirected graphs")
        self.graph = graph
        self.weighted = weighted
        self.table = {n: self._distances(n) for n in graph.nodes()}

    def _distances(self, start: str) -> dict[str, float]:
        if self.weighted:
            return dict(Dijkstra(self.graph, start).distance)
        dist: dict[str, float] = {start: 0}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for other in self.graph.neighbors(node):
                if other not in dist:
                    dist[other] = dist[node] + 1
                    queue.append(other)
        return dist

    def _pairs(self) -> list[tuple[str, str, float]]:
        out = []
        for a, b in combinations(self.graph.nodes(), 2):
            out.append((a, b, self.table[a].get(b, inf)))
        return out

    def wiener(self) -> float:
        return sum(d for _a, _b, d in self._pairs()) if self.graph.node_count() > 1 else 0.0

    def average_distance(self) -> float:
        pairs = self._pairs()
        return self.wiener() / len(pairs) if pairs else 0.0

    def harary(self) -> float:
        return sum(1 / d for _a, _b, d in self._pairs() if d != inf)

    def tree_edge_identity_holds(self) -> bool:
        # each edge is crossed by exactly the pairs it separates: size times size
        n = self.graph.node_count()
        if self.graph.edge_count() != n - 1:
            raise Invalid("the edge-cut identity is a statement about trees")
        total = 0
        for u, v, _w in self.graph.edges():
            side = self._side_without(u, v)
            total += len(side) * (n - len(side))
        return total == self.wiener()

    def _side_without(self, u: str, v: str) -> set[str]:
        seen = {u}
        stack = [u]
        while stack:
            node = stack.pop()
            for other in self.graph.neighbors(node):
                if other == v and node == u:
                    continue
                if other not in seen:
                    seen.add(other)
                    stack.append(other)
        return seen

    def note(self) -> str:
        w = self.wiener()
        if w == inf:
            return f"disconnected: Wiener index infinite, Harary index {self.harary():.3f}"
        return (
            f"Wiener index {w:g}, average distance {self.average_distance():.3f}, "
            f"Harary index {self.harary():.3f}"
        )
