"""Dijkstra: shortest paths from a source when no edge weight is negative.

Dijkstra's algorithm finds the shortest path from a source to every node in
a graph with non-negative edge weights. It grows a set of nodes whose
shortest distance is settled, always settling next the unsettled node with
the smallest tentative distance, and relaxing its edges: for each neighbor,
if reaching it through the just-settled node is shorter than its current
tentative distance, that shorter distance is recorded. The reason settling
the smallest tentative distance is safe, and the reason the whole algorithm
works, is that with no negative edges a path can only get longer as it
grows, so the smallest tentative distance can never later be improved by a
detour through a farther node. That single assumption is load-bearing: a
negative edge would let a longer-looking path turn out shorter after a
negative step, and Dijkstra would settle a node too early and report a
wrong distance, which is why a negative weight must be refused rather than
handled quietly. The engine uses a binary heap as the priority queue and
pushes a new entry on each improvement rather than decreasing a key in
place, which the heap does not support; stale entries are skipped when
popped by checking the distance they carry against the best known, a lazy
deletion that keeps the heap simple at the cost of a few extra entries. The
solver returns distances, reconstructs a path via parent pointers, refuses
a negative edge weight and a missing source, and reports the number of
nodes settled and the farthest settled distance, the weighted eccentricity
of the source, the reading a weighted diameter is built from.
"""

from __future__ import annotations

import heapq
import math

from mesh.errors import Invalid, Missing, Unreachable
from mesh.graph import Graph


class Dijkstra:
    def __init__(self, graph: Graph, source: str) -> None:
        if not graph.has_node(source):
            raise Missing(f"source '{source}' is not in the graph")
        self.graph = graph
        self.source = source
        self.distance: dict[str, float] = {source: 0.0}
        self.parent: dict[str, str | None] = {source: None}
        self._run()

    def _run(self) -> None:
        heap: list[tuple[float, str]] = [(0.0, self.source)]
        settled: set[str] = set()
        while heap:
            dist, node = heapq.heappop(heap)
            if node in settled:
                continue  # a stale heap entry, superseded by a shorter one
            settled.add(node)
            for neighbor, weight in self.graph.neighbors(node).items():
                if weight < 0:
                    raise Invalid(
                        f"edge {node}->{neighbor} has negative weight {weight}; "
                        "Dijkstra would settle a node too early, use Bellman-Ford"
                    )
                candidate = dist + weight
                if candidate < self.distance.get(neighbor, math.inf):
                    self.distance[neighbor] = candidate
                    self.parent[neighbor] = node
                    heapq.heappush(heap, (candidate, neighbor))

    def distance_to(self, node: str) -> float:
        if node not in self.distance:
            raise Unreachable(f"'{node}' is unreachable from '{self.source}'")
        return self.distance[node]

    def path_to(self, node: str) -> list[str]:
        if node not in self.distance:
            raise Unreachable(f"'{node}' is unreachable from '{self.source}'")
        path: list[str] = []
        current: str | None = node
        while current is not None:
            path.append(current)
            current = self.parent[current]
        path.reverse()
        return path

    def settled_count(self) -> int:
        return len(self.distance)

    def eccentricity(self) -> float:
        return max(self.distance.values())

    def note(self) -> str:
        return (
            f"settled {len(self.distance)} node(s) from '{self.source}', "
            f"farthest at weighted distance {self.eccentricity()}; safe only "
            "because no edge is negative"
        )
