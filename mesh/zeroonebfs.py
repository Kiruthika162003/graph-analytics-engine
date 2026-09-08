"""0-1 BFS: shortest paths when every edge costs zero or one, with no heap at all.

Dijkstra pays a logarithm per edge for its heap, and on a graph whose
weights are all zero or one that is more machinery than the problem
needs. A zero-weight edge does not increase the distance, so the node it
reaches belongs in the same ring as the current node; a one-weight edge
reaches the next ring. Breadth-first search already processes one ring
at a time, so the only change needed is to put a node reached by a
zero edge at the front of the queue, to be processed with the current
ring, and a node reached by a one edge at the back, to wait for the
next. A double-ended queue does both in constant time, and the search
runs in time linear in nodes plus edges, the same as plain BFS, with
distances that are exactly the shortest weighted paths. The
correctness argument is that the deque always holds nodes of at most
two consecutive distances, current at the front and current plus one at
the back, in order, so it behaves as the priority queue Dijkstra would
have used while never needing to sort. A node may be pushed more than
once, since a later zero edge can improve a distance already assigned
by a one edge, so the search relaxes on pop and skips stale entries the
way Dijkstra does. The problem shows up more than its statement
suggests: minimum number of edges to flip to make a path, grid mazes
with free moves in one direction, and any shortest path where the cost
is a count of some rare event. The search returns distances and paths,
refuses a weight that is not zero or one, and reports how many nodes
were pushed against how many exist, because a push count near the node
count is a graph where the deque ordering did its job and one far above
it is a graph with many zero edges reopening nodes already placed.
"""

from __future__ import annotations

from collections import deque

from mesh.errors import Invalid, Missing, Unreachable
from mesh.graph import Graph


class ZeroOneBFS:
    def __init__(self, graph: Graph, source: str) -> None:
        if not graph.has_node(source):
            raise Missing(f"source '{source}' is not in the graph")
        for u, v, w in graph.edges():
            if w not in (0, 1):
                raise Invalid(f"edge {u}->{v} has weight {w}; 0-1 BFS needs zero or one")
        self.graph = graph
        self.source = source
        self.distance: dict[str, int] = {source: 0}
        self.parent: dict[str, str | None] = {source: None}
        self.pushes = 0
        self._run()

    def _run(self) -> None:
        queue: deque[str] = deque([self.source])
        self.pushes = 1
        while queue:
            node = queue.popleft()
            for nbr, w in self.graph.neighbors(node).items():
                candidate = self.distance[node] + int(w)
                if candidate < self.distance.get(nbr, float("inf")):
                    self.distance[nbr] = candidate
                    self.parent[nbr] = node
                    self.pushes += 1
                    # zero edges join the current ring at the front, ones wait
                    if w == 0:
                        queue.appendleft(nbr)
                    else:
                        queue.append(nbr)

    def distance_to(self, node: str) -> int:
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

    def note(self) -> str:
        return (
            f"{self.pushes} push(es) over {self.graph.node_count()} node(s) with no "
            "heap; a count near the node count is the deque ordering doing its job"
        )
