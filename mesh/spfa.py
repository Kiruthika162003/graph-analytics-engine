"""SPFA: Bellman-Ford that only revisits the nodes whose distance just changed.

Bellman-Ford relaxes every edge in every pass, even the edges out of
nodes whose distance did not change since the last pass and so cannot
improve anything. The shortest path faster algorithm keeps a queue of
exactly the nodes whose distance was just lowered, pops one, relaxes its
outgoing edges, and pushes any neighbor it improved that is not already
queued. On most graphs this touches a small fraction of what full passes
would, and on a graph with no negative edges it behaves like a
breadth-first Dijkstra without the heap. The name promises more than
the algorithm delivers: its worst case is still nodes times edges, the
same as Bellman-Ford, and adversarial graphs make it hit that bound, so
the honest description is a good average case with an unchanged
guarantee. Negative-cycle detection comes from counting. A node that is
dequeued more than the node count many times is being relaxed around a
loop that keeps lowering it, which can only be a negative cycle, so the
algorithm refuses at that point rather than spinning forever. The count
threshold is what makes the detection sound: a shortest path has at most
nodes minus one edges, so no node on a real shortest path is improved
more than that many times. The solver returns distances and paths,
raises on a negative cycle naming the node that betrayed it, and reports
the total dequeue count against nodes times edges, because that ratio
is how far below the worst case the queue discipline kept the work, and
a ratio near one is a graph that defeated the heuristic.
"""

from __future__ import annotations

import math
from collections import deque

from mesh.errors import Missing, Negative, Unreachable
from mesh.graph import Graph


class SPFA:
    def __init__(self, graph: Graph, source: str) -> None:
        if not graph.has_node(source):
            raise Missing(f"source '{source}' is not in the graph")
        self.graph = graph
        self.source = source
        self.distance: dict[str, float] = dict.fromkeys(graph.nodes(), math.inf)
        self.distance[source] = 0.0
        self.parent: dict[str, str | None] = {source: None}
        self.dequeues = 0
        self._run()

    def _run(self) -> None:
        n = self.graph.node_count()
        queue: deque[str] = deque([self.source])
        queued = {self.source}
        times = dict.fromkeys(self.graph.nodes(), 0)
        while queue:
            u = queue.popleft()
            queued.discard(u)
            self.dequeues += 1
            times[u] += 1
            if times[u] > n:
                raise Negative(
                    f"'{u}' was dequeued more than {n} times; a negative cycle keeps "
                    "lowering it and no shortest path exists"
                )
            for v, w in self.graph.neighbors(u).items():
                candidate = self.distance[u] + w
                if candidate < self.distance[v]:
                    self.distance[v] = candidate
                    self.parent[v] = u
                    if v not in queued:
                        queue.append(v)  # only nodes that just improved rejoin
                        queued.add(v)

    def distance_to(self, node: str) -> float:
        if self.distance.get(node, math.inf) == math.inf:
            raise Unreachable(f"'{node}' is unreachable from '{self.source}'")
        return self.distance[node]

    def path_to(self, node: str) -> list[str]:
        if self.distance.get(node, math.inf) == math.inf:
            raise Unreachable(f"'{node}' is unreachable from '{self.source}'")
        path: list[str] = []
        current: str | None = node
        while current is not None:
            path.append(current)
            current = self.parent.get(current)
        path.reverse()
        return path

    def work_ratio(self) -> float:
        bound = self.graph.node_count() * max(1, self.graph.edge_count())
        return self.dequeues / bound

    def note(self) -> str:
        return (
            f"{self.dequeues} dequeue(s), {self.work_ratio() * 100:.1f}% of the "
            "nodes-times-edges bound; near 100 is a graph that defeated the heuristic"
        )
