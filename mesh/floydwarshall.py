"""Floyd-Warshall: all-pairs shortest paths by growing the set of waypoints.

Where Dijkstra and Bellman-Ford answer shortest paths from one source,
Floyd-Warshall answers them between every pair of nodes at once, and it does
so with a strikingly short idea. Consider the shortest path between two
nodes that is allowed to pass only through waypoints drawn from some set.
Add one more node to the allowed set. The shortest path either still does
not use the new node, in which case it is unchanged, or it does, in which
case it goes from the start to the new node and then from the new node to
the end, each of those two legs itself a shortest path through the smaller
set. So the algorithm considers each node in turn as a permitted waypoint
and, for every pair, asks whether routing through that waypoint beats the
best found so far. After every node has had its turn as a waypoint, every
pair holds its true shortest distance. Three nested loops, nodes cubed,
and the whole all-pairs table is filled. Floyd-Warshall tolerates negative
edges, like Bellman-Ford, and detects a negative cycle by the same tell: if
any node ends up with a negative distance to itself, a zero-length round
trip that came out below zero, then a negative cycle passes through it. The
solver builds the distance table and a next-hop table for path
reconstruction, reports a pair's distance and path, raises on a negative
self-distance, and reports the graph's diameter, the largest finite
all-pairs distance, the longest shortest path, which a single-source run
cannot give without being run from every node.
"""

from __future__ import annotations

import math

from mesh.errors import Missing, Negative, Unreachable
from mesh.graph import Graph


class FloydWarshall:
    def __init__(self, graph: Graph) -> None:
        self.graph = graph
        nodes = graph.nodes()
        self.dist: dict[str, dict[str, float]] = {
            u: dict.fromkeys(nodes, math.inf) for u in nodes
        }
        self.nxt: dict[str, dict[str, str | None]] = {
            u: dict.fromkeys(nodes, None) for u in nodes
        }
        for u in nodes:
            self.dist[u][u] = 0.0
        for u, v, w in graph.edges():
            self.dist[u][v] = w
            self.nxt[u][v] = v
            if not graph.directed:
                # edges() lists an undirected edge once; the postman test caught
                # the reverse direction being left at infinity
                self.dist[v][u] = w
                self.nxt[v][u] = u
        self._run(nodes)

    def _run(self, nodes: list[str]) -> None:
        for k in nodes:
            for i in nodes:
                if self.dist[i][k] == math.inf:
                    continue  # i cannot reach the waypoint, no route through it
                for j in nodes:
                    through = self.dist[i][k] + self.dist[k][j]
                    if through < self.dist[i][j]:
                        self.dist[i][j] = through
                        self.nxt[i][j] = self.nxt[i][k]
        for u in nodes:
            if self.dist[u][u] < 0:
                raise Negative(
                    f"'{u}' has a negative distance to itself; a negative cycle "
                    "passes through it and no shortest path exists"
                )

    def distance(self, u: str, v: str) -> float:
        if u not in self.dist or v not in self.dist:
            raise Missing("both endpoints must be in the graph")
        if self.dist[u][v] == math.inf:
            raise Unreachable(f"'{v}' is unreachable from '{u}'")
        return self.dist[u][v]

    def path(self, u: str, v: str) -> list[str]:
        if self.dist[u][v] == math.inf:
            raise Unreachable(f"'{v}' is unreachable from '{u}'")
        path = [u]
        current = u
        while current != v:
            current = self.nxt[current][v]  # type: ignore[assignment]
            path.append(current)
        return path

    def diameter(self) -> float:
        finite = [
            self.dist[u][v]
            for u in self.dist
            for v in self.dist[u]
            if self.dist[u][v] != math.inf
        ]
        return max(finite) if finite else 0.0

    def note(self) -> str:
        return (
            f"all-pairs table filled for {self.graph.node_count()} node(s), "
            f"diameter {self.diameter()}; the diameter is the longest shortest "
            "path, which a single-source run cannot give alone"
        )
