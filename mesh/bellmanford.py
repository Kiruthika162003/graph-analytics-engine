"""Bellman-Ford: shortest paths that tolerate negative edges and catch the loop.

Bellman-Ford solves the single-source shortest-path problem where Dijkstra
cannot: when some edge weights are negative. It does not need the ordering
trick that Dijkstra relies on, so it does not need non-negative weights.
Instead it relaxes every edge in the graph, then does it again, and again,
a number of times one less than the number of nodes. The reason that count
is exactly right is that a shortest path in a graph with no negative cycle
visits each node at most once, so it has at most nodes-minus-one edges, and
each full pass of relaxation is guaranteed to extend every shortest path by
at least one more correct edge. After nodes-minus-one passes every shortest
path is fully built. That gives the algorithm its power and its cost: it is
slower than Dijkstra, edges times nodes rather than edges times a logarithm,
the price of not assuming non-negative weights. The same structure detects
the one thing that makes shortest paths meaningless: a negative cycle. If
one more relaxation pass, the nth, still improves some distance, then a
path is getting shorter without bound by going around a negative-weight
loop, and no shortest distance exists. Bellman-Ford reports that rather
than looping forever or returning a nonsense finite number. The solver
returns distances, reconstructs paths, raises on a negative cycle naming a
node on it, and reports the pass at which distances stopped changing, an
early-exit reading, because a graph whose distances settle well before the
nodes-minus-one bound has short paths and the later passes were wasted work.
"""

from __future__ import annotations

import math

from mesh.errors import Missing, Negative, Unreachable
from mesh.graph import Graph


class BellmanFord:
    def __init__(self, graph: Graph, source: str) -> None:
        if not graph.has_node(source):
            raise Missing(f"source '{source}' is not in the graph")
        self.graph = graph
        self.source = source
        self.distance: dict[str, float] = dict.fromkeys(graph.nodes(), math.inf)
        self.distance[source] = 0.0
        self.parent: dict[str, str | None] = {source: None}
        self.settled_pass = 0
        self._run()

    def _run(self) -> None:
        edges = self.graph.edges()
        node_count = self.graph.node_count()
        for i in range(node_count - 1):
            changed = False
            for u, v, w in edges:
                if self.distance[u] + w < self.distance[v]:
                    self.distance[v] = self.distance[u] + w
                    self.parent[v] = u
                    changed = True
            if not changed:
                # a full pass with no improvement means every path is built
                self.settled_pass = i + 1
                break
        else:
            self.settled_pass = node_count - 1
        # one more pass: any improvement now betrays a negative cycle
        for u, v, w in edges:
            if self.distance[u] + w < self.distance[v]:
                raise Negative(
                    f"a negative cycle reaches '{v}'; distances shrink without "
                    "bound around it, so no shortest path exists"
                )

    def distance_to(self, node: str) -> float:
        if node not in self.distance or self.distance[node] == math.inf:
            raise Unreachable(f"'{node}' is unreachable from '{self.source}'")
        return self.distance[node]

    def path_to(self, node: str) -> list[str]:
        if node not in self.distance or self.distance[node] == math.inf:
            raise Unreachable(f"'{node}' is unreachable from '{self.source}'")
        path: list[str] = []
        current: str | None = node
        while current is not None:
            path.append(current)
            current = self.parent.get(current)
        path.reverse()
        return path

    def note(self) -> str:
        return (
            f"distances settled after pass {self.settled_pass} of at most "
            f"{self.graph.node_count() - 1}; settling early means short paths "
            "and the later passes would have been wasted work"
        )
