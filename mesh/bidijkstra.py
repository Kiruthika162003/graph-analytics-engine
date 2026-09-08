"""Bidirectional Dijkstra: two weighted searches meet, and a subtle rule says when.

Running Dijkstra from the source and from the target at once, on the
reverse graph for the backward side, halves the radius each search must
cover and, on a graph with real branching, cuts the settled-node count
sharply, the same saving bidirectional BFS gets. The weighted version
has a trap that the unweighted one does not. When the two searches first
touch, at some node settled by both, the path through that node is not
necessarily shortest: a cheaper path may cross between the frontiers
along an edge whose endpoints are each still in one side's heap,
neither yet settled. So the search keeps the best combined distance
seen over every edge relaxed across the frontier, forward distance to
one end plus edge weight plus backward distance from the other, and it
stops only when the smallest key still in the forward heap plus the
smallest key in the backward heap is at least that best, because at
that point no unsettled pair can beat it. Stopping at the first common
settled node instead is the classic mistake, and it returns a wrong
distance on graphs with a cheap crossing edge. The engine alternates
sides by taking whichever heap has the smaller top, tracks the meeting
edge that achieved the best distance, and reconstructs the path by
joining the forward parent chain to one end with the backward parent
chain from the other. It refuses a negative edge for Dijkstra's reason
and reports the settled count against what one-sided Dijkstra settled to
reach the target, because that ratio is the saving, and a ratio near one
is a long thin graph where meeting in the middle buys nothing.
"""

from __future__ import annotations

import heapq
import math

from mesh.dijkstra import Dijkstra
from mesh.errors import Invalid, Missing, Unreachable
from mesh.graph import Graph


class BidirectionalDijkstra:
    def __init__(self, graph: Graph, source: str, target: str) -> None:
        for n in (source, target):
            if not graph.has_node(n):
                raise Missing(f"'{n}' is not in the graph")
        for u, v, w in graph.edges():
            if w < 0:
                raise Invalid(f"edge {u}->{v} is negative; Dijkstra needs non-negative weights")
        self.graph = graph
        self.source = source
        self.target = target
        self.backward = graph.reverse() if graph.directed else graph
        self.settled = 0
        self.best = math.inf
        self.meeting: tuple[str, str] | None = None
        self._dist_f: dict[str, float] = {source: 0.0}
        self._dist_b: dict[str, float] = {target: 0.0}
        self._par_f: dict[str, str | None] = {source: None}
        self._par_b: dict[str, str | None] = {target: None}
        self._run()

    def _run(self) -> None:
        if self.source == self.target:
            self.best = 0.0
            self.meeting = (self.source, self.source)
            return
        heap_f = [(0.0, self.source)]
        heap_b = [(0.0, self.target)]
        done_f: set[str] = set()
        done_b: set[str] = set()
        while heap_f and heap_b:
            # stop once no unsettled pair could beat the best crossing seen
            if heap_f[0][0] + heap_b[0][0] >= self.best:
                break
            if heap_f[0][0] <= heap_b[0][0]:
                self._step(
                    heap_f, done_f, self.graph, self._dist_f, self._par_f, self._dist_b, True
                )
            else:
                self._step(
                    heap_b, done_b, self.backward,
                    self._dist_b, self._par_b, self._dist_f, False,
                )

    def _step(
        self,
        heap: list[tuple[float, str]],
        done: set[str],
        graph: Graph,
        dist: dict[str, float],
        parent: dict[str, str | None],
        other: dict[str, float],
        forward: bool,
    ) -> None:
        d, node = heapq.heappop(heap)
        if node in done:
            return
        done.add(node)
        self.settled += 1
        for nbr, w in graph.neighbors(node).items():
            candidate = d + w
            if candidate < dist.get(nbr, math.inf):
                dist[nbr] = candidate
                parent[nbr] = node
                heapq.heappush(heap, (candidate, nbr))
            if nbr in other and candidate + other[nbr] < self.best:
                # a crossing edge: forward end to backward end through nbr
                self.best = candidate + other[nbr]
                self.meeting = (node, nbr) if forward else (nbr, node)

    def distance(self) -> float:
        if self.meeting is None:
            raise Unreachable(f"'{self.target}' is unreachable from '{self.source}'")
        return self.best

    def path(self) -> list[str]:
        if self.meeting is None:
            raise Unreachable(f"'{self.target}' is unreachable from '{self.source}'")
        left_end, right_end = self.meeting
        left: list[str] = []
        cur: str | None = left_end
        while cur is not None:
            left.append(cur)
            cur = self._par_f[cur]
        left.reverse()
        right: list[str] = []
        cur = right_end
        while cur is not None:
            right.append(cur)
            cur = self._par_b[cur]
        if left_end == right_end:
            right = right[1:]
        return left + right

    def one_sided_settled(self) -> int:
        run = Dijkstra(self.graph, self.source)
        if self.target not in run.distance:
            return run.settled_count()
        limit = run.distance[self.target]
        return sum(1 for d in run.distance.values() if d <= limit)

    def note(self) -> str:
        return (
            f"settled {self.settled} node(s) against {self.one_sided_settled()} one-sided; "
            "the stop rule is heap tops summing past the best crossing, not the first "
            "common node"
        )
