"""DAG paths: shortest and longest paths in one topological sweep, negatives welcome.

When a directed graph has no cycles, shortest paths become easy in a way
they are not in general. Process the nodes in topological order, and by the
time a node is reached every path into it has already been fully relaxed,
because all of its predecessors come earlier in the order. So a single pass
over the nodes in that order, relaxing each node's outgoing edges once,
computes the shortest distance from a source to every node in time linear
in nodes plus edges, faster than Dijkstra's heap, and it tolerates negative
edge weights without any of Bellman-Ford's repeated passes, since a
negative cycle is impossible in a graph with no cycles at all. The same
sweep, with every comparison flipped from less-than to greater-than,
computes longest paths, which is the problem a general graph cannot solve
efficiently but a DAG can, and it is the one that matters for scheduling:
the longest path through a dependency graph, where edge weights are task
durations, is the critical path, the minimum time the whole project can
take, and every task on it is one whose delay delays the finish. The solver
takes a topological order from the sort, refusing a cyclic graph through
it, then runs the forward relaxation once for shortest and once for
longest, reconstructing paths through parent pointers. It reports the
critical path length and the tasks on it, because that path is what a
scheduler is really asking for when it asks about a DAG, and a critical
path that runs through almost every node is a project with no slack
anywhere, where nothing can slip.
"""

from __future__ import annotations

import math

from mesh.errors import Missing, Unreachable
from mesh.graph import Graph
from mesh.toposort import TopologicalSort


class DagPaths:
    def __init__(self, graph: Graph, source: str) -> None:
        if not graph.has_node(source):
            raise Missing(f"source '{source}' is not in the graph")
        self.graph = graph
        self.source = source
        self.order = TopologicalSort(graph).order()  # refuses a cycle
        self.shortest: dict[str, float] = {}
        self.longest: dict[str, float] = {}
        self._short_parent: dict[str, str | None] = {}
        self._long_parent: dict[str, str | None] = {}
        self._sweep(shortest=True)
        self._sweep(shortest=False)

    def _sweep(self, shortest: bool) -> None:
        # one forward pass in topological order relaxes every path exactly once
        start = math.inf if shortest else -math.inf
        dist = dict.fromkeys(self.graph.nodes(), start)
        parent: dict[str, str | None] = {self.source: None}
        dist[self.source] = 0.0
        for node in self.order:
            if dist[node] == start:
                continue  # not reached from the source yet
            for nbr, w in self.graph.neighbors(node).items():
                candidate = dist[node] + w
                better = candidate < dist[nbr] if shortest else candidate > dist[nbr]
                if better:
                    dist[nbr] = candidate
                    parent[nbr] = node
        if shortest:
            self.shortest, self._short_parent = dist, parent
        else:
            self.longest, self._long_parent = dist, parent

    def _path(self, node: str, parent: dict[str, str | None]) -> list[str]:
        if node not in parent:
            raise Unreachable(f"'{node}' is unreachable from '{self.source}'")
        path: list[str] = []
        current: str | None = node
        while current is not None:
            path.append(current)
            current = parent[current]
        path.reverse()
        return path

    def shortest_to(self, node: str) -> float:
        if self.shortest.get(node, math.inf) == math.inf:
            raise Unreachable(f"'{node}' is unreachable from '{self.source}'")
        return self.shortest[node]

    def longest_to(self, node: str) -> float:
        if self.longest.get(node, -math.inf) == -math.inf:
            raise Unreachable(f"'{node}' is unreachable from '{self.source}'")
        return self.longest[node]

    def shortest_path_to(self, node: str) -> list[str]:
        return self._path(node, self._short_parent)

    def critical_path(self) -> list[str]:
        reached = {n: d for n, d in self.longest.items() if d != -math.inf}
        # ties break toward the later node in topological order: a zero-weight
        # final edge ties the sink with its predecessor, and the sink is the end.
        # First guess broke ties alphabetically, which stopped one node short.
        rank = {n: i for i, n in enumerate(self.order)}
        end = max(reached, key=lambda n: (reached[n], rank[n]))
        return self._path(end, self._long_parent)

    def note(self) -> str:
        path = self.critical_path()
        return (
            f"critical path of length {self.longest[path[-1]]} through "
            f"{len(path)} of {self.graph.node_count()} node(s); a path through "
            "nearly every node is a project with no slack anywhere"
        )
