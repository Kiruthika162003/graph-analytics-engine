"""Graph center and periphery: the nodes closest to everyone, and the ones farthest.

The eccentricity of a node is the longest shortest path from it to
any other node, the radius is the smallest eccentricity, the diameter
the largest, and the center is the set of nodes that reach the radius
while the periphery reaches the diameter. A warehouse should sit at a
center node and a broadcast should start there; a periphery node is
the one that hears last. Two facts pin the numbers: the diameter is at
most twice the radius, since any two nodes can route through a center
node, and a tree's center is one node or two adjacent nodes, found by
Jordan's peeling, which strips every leaf in rounds until one or two
nodes remain. The engine computes every eccentricity by a breadth-first
sweep from each node in hops, or by Dijkstra distances when asked for
weights, reads radius, diameter, center, and periphery, checks the
doubling bound, and on a tree runs the peeling and confirms it lands
on the same center the eccentricities name, which ties a global
computation to a local one. A disconnected graph has infinite
eccentricities, which the module reports as such rather than pretending
a finite radius; a directed graph is refused because eccentricity here
means an undirected distance.
"""

from __future__ import annotations

from collections import deque
from math import inf

from mesh.dijkstra import Dijkstra
from mesh.errors import Invalid
from mesh.graph import Graph


class GraphCenter:
    def __init__(self, graph: Graph, weighted: bool = False) -> None:
        if graph.directed:
            raise Invalid("center and periphery are read on an undirected graph")
        self.graph = graph
        self.weighted = weighted
        self.eccentricity = {n: self._eccentricity(n) for n in graph.nodes()}

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

    def _eccentricity(self, node: str) -> float:
        dist = self._distances(node)
        if len(dist) < self.graph.node_count():
            return inf
        return max(dist.values())

    def radius(self) -> float:
        return min(self.eccentricity.values(), default=0)

    def diameter(self) -> float:
        return max(self.eccentricity.values(), default=0)

    def center(self) -> list[str]:
        r = self.radius()
        return sorted(n for n, e in self.eccentricity.items() if e == r)

    def periphery(self) -> list[str]:
        d = self.diameter()
        return sorted(n for n, e in self.eccentricity.items() if e == d)

    def doubling_bound_holds(self) -> bool:
        d, r = self.diameter(), self.radius()
        return d == inf or d <= 2 * r

    def peel_tree(self) -> list[str]:
        # Jordan's peeling: remove every leaf each round until one or two nodes remain
        if self.graph.edge_count() != self.graph.node_count() - 1:
            raise Invalid("peeling needs a tree, which has exactly n minus one edges")
        degree = {n: self.graph.degree(n) for n in self.graph.nodes()}
        remaining = set(degree)
        while len(remaining) > 2:
            leaves = [n for n in remaining if degree[n] <= 1]
            if not leaves:
                raise Invalid("peeling found no leaf, so this is not a tree")
            for leaf in leaves:
                remaining.discard(leaf)
                for m in self.graph.neighbors(leaf):
                    if m in remaining:
                        degree[m] -= 1
        return sorted(remaining)

    def note(self) -> str:
        unit = "weight" if self.weighted else "hop(s)"
        if self.radius() == inf:
            return "disconnected: every eccentricity is infinite, so there is no center"
        return (
            f"radius {self.radius()} and diameter {self.diameter()} in {unit}; center "
            f"{self.center()}, periphery {self.periphery()}"
        )
