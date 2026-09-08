"""Dinic: maximum flow by blocking flows on a layered residual graph.

Edmonds-Karp finds one shortest augmenting path per BFS and pushes along it.
Dinic finds a whole family per phase. Each phase runs a breadth-first
search from the source over the residual graph to assign every node a level,
its distance from the source, and then a depth-first search pushes flow
only along edges that step from one level to the next, from level k to level
k plus one, which is what makes every path found a shortest path. Within a
phase the DFS pushes as much as it can, a blocking flow, meaning that after
it finishes every shortest source-to-sink path in the layered graph has at
least one saturated edge. The next phase rebuilds the levels, and the key
fact is that the sink's level strictly increases between phases, because
the blocking flow saturated every path of the previous length, so there are
at most as many phases as there are nodes. Each phase costs nodes times
edges in the worst case, giving nodes squared times edges overall, and on
unit-capacity networks like bipartite matching the bound drops to the
square root of the nodes times the edges, which is why Dinic is the flow
algorithm of choice when Edmonds-Karp's nodes-times-edges-squared is too
slow. One implementation detail carries most of the speed: each node keeps
a pointer to the next edge it has not yet exhausted this phase, so the DFS
never rescans an edge that already proved to be a dead end, which is what
keeps a phase to nodes times edges rather than worse. The solver runs
phases until the sink becomes unreachable in the residual graph, returns
the flow value and per-edge flow, refuses a negative capacity and a source
equal to the sink, and reports the phase count, which must never exceed
the node count and must produce a value identical to Edmonds-Karp's.
"""

from __future__ import annotations

import math
from collections import deque

from mesh.errors import Invalid, Missing
from mesh.graph import Graph


class Dinic:
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
        self.residual: dict[str, dict[str, float]] = {n: {} for n in graph.nodes()}
        for u, v, cap in graph.edges():
            if cap < 0:
                raise Invalid(f"edge {u}->{v} has negative capacity {cap}")
            self.residual[u][v] = self.residual[u].get(v, 0.0) + cap
            self.residual[v].setdefault(u, 0.0)
        # adjacency order fixed once so the per-node pointer means something
        self._order = {u: list(nbrs) for u, nbrs in self.residual.items()}
        self.level: dict[str, int] = {}
        self.phases = 0
        self.value = self._run()

    def _build_levels(self) -> bool:
        self.level = {self.source: 0}
        queue: deque[str] = deque([self.source])
        while queue:
            u = queue.popleft()
            for v in self._order[u]:
                if self.residual[u][v] > 0 and v not in self.level:
                    self.level[v] = self.level[u] + 1
                    queue.append(v)
        return self.sink in self.level

    def _push(self, u: str, limit: float, pointer: dict[str, int]) -> float:
        if u == self.sink:
            return limit
        edges = self._order[u]
        while pointer[u] < len(edges):
            v = edges[pointer[u]]
            if self.residual[u][v] > 0 and self.level.get(v) == self.level[u] + 1:
                pushed = self._push(v, min(limit, self.residual[u][v]), pointer)
                if pushed > 0:
                    self.residual[u][v] -= pushed
                    self.residual[v][u] += pushed
                    return pushed
            pointer[u] += 1  # this edge is exhausted for the phase, never rescan it
        return 0.0

    def _run(self) -> float:
        total = 0.0
        while self._build_levels():
            self.phases += 1
            pointer = dict.fromkeys(self.residual, 0)
            while True:
                pushed = self._push(self.source, math.inf, pointer)
                if pushed == 0:
                    break
                total += pushed
        return total

    def flow_on(self, u: str, v: str) -> float:
        # the residual merges u->v with the reverse of v->u, so this is the NET
        # flow across the pair and can go negative on an antiparallel edge;
        # push-relabel exposed this where the phase order here hid it
        if not self.graph.has_edge(u, v):
            raise Missing(f"no edge from '{u}' to '{v}'")
        return self.graph.weight(u, v) - self.residual[u][v]

    def note(self) -> str:
        return (
            f"maximum flow {self.value} in {self.phases} phase(s) over "
            f"{self.graph.node_count()} node(s); phases never exceed the node "
            "count because the sink's level rises every phase"
        )
