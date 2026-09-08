"""Breadth-first search: explore in rings, and the first time you reach a node
is along a shortest unweighted path.

Breadth-first search visits a graph in rings around a source: first the
source, then everything one edge away, then everything two edges away, and
so on, never starting on a farther ring before the nearer one is finished.
That order is the whole reason BFS matters. Because it reaches nodes in
nondecreasing order of distance, the first time it arrives at a node it has
arrived along a path with the fewest edges, so BFS computes shortest paths
on an unweighted graph for free, as a side effect of the order it visits
in. A depth-first search would reach the same nodes but not by any
shortest path, which is the difference the two make. The engine keeps a
queue of the frontier and a distance for each discovered node, dequeues a
node, and for each undiscovered neighbor records its distance as one more
than the current node's and its parent as the current node, then enqueues
it. A node is recorded as discovered when it is enqueued, not when it is
dequeued, because recording it at dequeue time would let it be enqueued
twice from two different neighbors on the same ring and inflate the work.
The parent pointers form a tree, the BFS tree, and following them from a
target back to the source reconstructs the shortest path. The search
returns distances from the source, reconstructs a path to any reached
node, and reports the set of reachable nodes, and it refuses a source that
is not in the graph and a path request to a node the search never reached.
It reports the number of nodes reached and the greatest distance, the
eccentricity of the source within its component, the reading a diameter
estimate is built from.
"""

from __future__ import annotations

from collections import deque

from mesh.errors import Missing, Unreachable
from mesh.graph import Graph


class BFS:
    def __init__(self, graph: Graph, source: str) -> None:
        if not graph.has_node(source):
            raise Missing(f"source '{source}' is not in the graph")
        self.graph = graph
        self.source = source
        self.distance: dict[str, int] = {source: 0}
        self.parent: dict[str, str | None] = {source: None}
        self._run()

    def _run(self) -> None:
        queue: deque[str] = deque([self.source])
        while queue:
            node = queue.popleft()
            for neighbor in self.graph.neighbors(node):
                if neighbor not in self.distance:
                    # discovered at enqueue time so it is never queued twice
                    self.distance[neighbor] = self.distance[node] + 1
                    self.parent[neighbor] = node
                    queue.append(neighbor)

    def reached(self, node: str) -> bool:
        return node in self.distance

    def distance_to(self, node: str) -> int:
        if node not in self.distance:
            raise Unreachable(f"'{node}' was not reached from '{self.source}'")
        return self.distance[node]

    def path_to(self, node: str) -> list[str]:
        if node not in self.distance:
            raise Unreachable(f"'{node}' was not reached from '{self.source}'")
        path: list[str] = []
        current: str | None = node
        while current is not None:
            path.append(current)
            current = self.parent[current]
        path.reverse()
        return path

    def reachable_nodes(self) -> set[str]:
        return set(self.distance)

    def eccentricity(self) -> int:
        # the greatest distance to any reached node, within the source component
        return max(self.distance.values())

    def note(self) -> str:
        return (
            f"reached {len(self.distance)} node(s) from '{self.source}', "
            f"farthest at distance {self.eccentricity()}; the first arrival at "
            "each node is along a shortest unweighted path"
        )
