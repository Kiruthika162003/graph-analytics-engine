"""A star: Dijkstra steered toward the goal by a heuristic that must not lie high.

A star finds a shortest path from a source to a single goal, and it is
Dijkstra with a sense of direction. Dijkstra expands nodes in order of the
distance already travelled, spreading outward blindly in every direction.
A star instead expands nodes in order of that distance plus an estimate of
the distance still remaining to the goal, a heuristic, so it pushes toward
the goal and leaves aside the nodes that lead away from it. When the
heuristic is good, this expands far fewer nodes than Dijkstra while
returning the same shortest path. The correctness hinges on one property
of the heuristic: it must be admissible, meaning it never overestimates the
true remaining distance. An admissible heuristic can undershoot freely,
even a constant zero, which just turns A star back into Dijkstra, but the
moment it overestimates it can make A star settle the goal through a path
that is not actually shortest, because the inflated estimate hid a better
route behind a wall of falsely-large scores. So the guarantee A star offers
is conditional on the heuristic told to it, and this engine states that
condition rather than assuming it: a zero heuristic is always safe and is
the default, and the caller supplying a heuristic owns its admissibility.
The search uses a heap keyed by travelled-plus-estimated cost, relaxes
edges the way Dijkstra does, refuses a negative edge for the same reason,
and stops as soon as the goal is settled rather than settling the whole
graph. It returns the path and its cost, reports how many nodes it expanded,
and compares that against the node count so the heuristic's steering shows
up as the fraction of the graph it managed to avoid touching.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable

from mesh.errors import Invalid, Missing, Unreachable
from mesh.graph import Graph


class AStar:
    def __init__(
        self,
        graph: Graph,
        source: str,
        goal: str,
        heuristic: Callable[[str], float] | None = None,
    ) -> None:
        if not graph.has_node(source):
            raise Missing(f"source '{source}' is not in the graph")
        if not graph.has_node(goal):
            raise Missing(f"goal '{goal}' is not in the graph")
        self.graph = graph
        self.source = source
        self.goal = goal
        # a zero heuristic is always admissible and reduces A star to Dijkstra
        self.heuristic = heuristic or (lambda _node: 0.0)
        self.distance: dict[str, float] = {source: 0.0}
        self.parent: dict[str, str | None] = {source: None}
        self.expanded = 0
        self._found = self._run()

    def _run(self) -> bool:
        start_f = self.heuristic(self.source)
        heap: list[tuple[float, str]] = [(start_f, self.source)]
        settled: set[str] = set()
        while heap:
            _f, node = heapq.heappop(heap)
            if node in settled:
                continue
            settled.add(node)
            self.expanded += 1
            if node == self.goal:
                return True  # stop as soon as the goal is settled
            for neighbor, weight in self.graph.neighbors(node).items():
                if weight < 0:
                    raise Invalid(
                        f"edge {node}->{neighbor} is negative; A star, like "
                        "Dijkstra, needs non-negative weights"
                    )
                candidate = self.distance[node] + weight
                if candidate < self.distance.get(neighbor, math.inf):
                    self.distance[neighbor] = candidate
                    self.parent[neighbor] = node
                    priority = candidate + self.heuristic(neighbor)
                    heapq.heappush(heap, (priority, neighbor))
        return False

    def cost(self) -> float:
        if not self._found:
            raise Unreachable(f"'{self.goal}' is unreachable from '{self.source}'")
        return self.distance[self.goal]

    def path(self) -> list[str]:
        if not self._found:
            raise Unreachable(f"'{self.goal}' is unreachable from '{self.source}'")
        path: list[str] = []
        current: str | None = self.goal
        while current is not None:
            path.append(current)
            current = self.parent[current]
        path.reverse()
        return path

    def note(self) -> str:
        avoided = self.graph.node_count() - self.expanded
        return (
            f"expanded {self.expanded} of {self.graph.node_count()} node(s), "
            f"avoided {avoided}; a good admissible heuristic is what turns those "
            "avoided nodes into skipped work"
        )
