"""Edmonds-Karp: maximum flow by shortest augmenting paths in the residual graph.

A flow network is a directed graph whose edges carry capacities, with a
source that produces flow and a sink that absorbs it, and the maximum flow
question asks how much can be pushed from source to sink without any edge
carrying more than its capacity and with every other node passing on
exactly what it receives. Ford-Fulkerson answers it by repeatedly finding
an augmenting path, a route from source to sink along which more flow can
still be sent, and pushing the bottleneck amount along it, until no such
path remains. The device that makes this correct is the residual graph:
each edge keeps a residual capacity, what it can still carry, and each unit
of flow pushed forward also opens a reverse edge of that amount, so a later
path can cancel flow already sent if a better routing exists. Without the
reverse edges the greedy pushes can paint themselves into a corner short of
the true maximum. Edmonds-Karp is Ford-Fulkerson with one rule: find each
augmenting path by breadth-first search, so it is a shortest path in edge
count. That rule bounds the number of augmentations to nodes times edges,
because each augmentation saturates a bottleneck edge and shortest-path
augmentation forces bottleneck distances to only ever grow, where an
arbitrary choice of path can take exponentially many steps or, with
irrational capacities, never terminate. The solver builds the residual
graph, augments along BFS paths until none is left, returns the maximum
flow value and the flow on each edge, and refuses a negative capacity and a
source that equals the sink. It reports the augmentation count against the
flow value, because many small augmentations for a modest flow is a network
with narrow parallel routes, the shape that makes the shortest-path rule
earn its keep.
"""

from __future__ import annotations

from collections import deque
from itertools import pairwise

from mesh.errors import Invalid, Missing
from mesh.graph import Graph


class EdmondsKarp:
    def __init__(self, graph: Graph, source: str, sink: str) -> None:
        if not graph.directed:
            raise Invalid("a flow network is a directed graph")
        if source == sink:
            raise Invalid("the source and sink must be different nodes")
        for endpoint in (source, sink):
            if not graph.has_node(endpoint):
                raise Missing(f"'{endpoint}' is not in the graph")
        self.graph = graph
        self.source = source
        self.sink = sink
        # residual[u][v] is what u can still send to v, reverse edges included
        self.residual: dict[str, dict[str, float]] = {n: {} for n in graph.nodes()}
        for u, v, cap in graph.edges():
            if cap < 0:
                raise Invalid(f"edge {u}->{v} has negative capacity {cap}")
            self.residual[u][v] = self.residual[u].get(v, 0.0) + cap
            self.residual[v].setdefault(u, 0.0)
        self.augmentations = 0
        self.value = self._run()

    def _bfs_path(self) -> list[str] | None:
        parent: dict[str, str | None] = {self.source: None}
        queue: deque[str] = deque([self.source])
        while queue:
            node = queue.popleft()
            for nbr, cap in self.residual[node].items():
                if cap > 0 and nbr not in parent:
                    parent[nbr] = node
                    if nbr == self.sink:
                        path = [nbr]
                        while parent[path[-1]] is not None:
                            path.append(parent[path[-1]])  # type: ignore[arg-type]
                        path.reverse()
                        return path
                    queue.append(nbr)
        return None

    def _run(self) -> float:
        total = 0.0
        while True:
            path = self._bfs_path()
            if path is None:
                return total
            bottleneck = min(
                self.residual[u][v] for u, v in pairwise(path)
            )
            for u, v in pairwise(path):
                # push forward and open the reverse edge for later cancellation
                self.residual[u][v] -= bottleneck
                self.residual[v][u] += bottleneck
            total += bottleneck
            self.augmentations += 1

    def flow_on(self, u: str, v: str) -> float:
        if not self.graph.has_edge(u, v):
            raise Missing(f"no edge from '{u}' to '{v}'")
        return self.graph.weight(u, v) - self.residual[u][v]

    def note(self) -> str:
        return (
            f"maximum flow {self.value} in {self.augmentations} augmentation(s); "
            "many small augmentations for a modest flow is a network of narrow "
            "parallel routes, the shape where shortest-path augmenting matters"
        )
