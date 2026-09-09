"""Bandwidth: how far apart an ordering forces the two ends of the worst edge.

Lay the nodes out in a line, and every edge spans some distance along
it. The bandwidth of the layout is the longest span, and the bandwidth
of the graph is the smallest such longest span over every layout. It
is the quantity a banded matrix solver cares about, since a sparse
matrix whose graph has bandwidth b can be stored and factored in
b-wide bands, and it is why the Cuthill-McKee ordering exists. Finding
the true bandwidth is NP-hard in general, so the engine gives two
readings. The upper reading is the bandwidth of a level-structure
ordering: breadth-first from a peripheral node, each level sorted by
degree, which is the Cuthill-McKee idea, taken from the best of a few
start nodes. The lower reading is the density bound: any layout that
holds a node and its k neighbors has some edge spanning at least the
ceiling of k over 2, so half the maximum degree rounded up is a floor,
and so is the ceiling of (n minus 1) over the diameter, because a
diameter path of d hops spans n minus 1 positions in a line. On small
graphs the exact value comes from trying every ordering, and the tests
check the two readings bracket it. A path has bandwidth one and a
complete graph n minus 1, which is where the two bounds meet; the note
states both readings and whether they meet.
"""

from __future__ import annotations

from collections import deque
from itertools import permutations
from math import ceil

from mesh.errors import Invalid
from mesh.graph import Graph


class Bandwidth:
    def __init__(self, graph: Graph) -> None:
        if graph.directed:
            raise Invalid("bandwidth is read on an undirected graph")
        self.graph = graph

    @staticmethod
    def span(graph: Graph, order: list[str]) -> int:
        position = {n: i for i, n in enumerate(order)}
        return max((abs(position[u] - position[v]) for u, v, _w in graph.edges()), default=0)

    def _levels(self, start: str) -> list[str]:
        # breadth-first, each level in rising degree order, unreached nodes appended
        order = [start]
        seen = {start}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            nbrs = sorted(
                (m for m in self.graph.neighbors(node) if m not in seen),
                key=lambda m: (self.graph.degree(m), m),
            )
            for m in nbrs:
                seen.add(m)
                order.append(m)
                queue.append(m)
        order.extend(n for n in self.graph.nodes() if n not in seen)
        return order

    def _hops(self, start: str) -> dict[str, int]:
        dist = {start: 0}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for m in self.graph.neighbors(node):
                if m not in dist:
                    dist[m] = dist[node] + 1
                    queue.append(m)
        return dist

    def upper(self) -> tuple[int, list[str]]:
        nodes = self.graph.nodes()
        if not nodes:
            return 0, []
        # the low-degree nodes are the usual peripheral starts; try the lowest few
        starts = sorted(nodes, key=lambda n: (self.graph.degree(n), n))[:4]
        best_order = self._levels(starts[0])
        best = self.span(self.graph, best_order)
        for start in starts[1:]:
            order = self._levels(start)
            width = self.span(self.graph, order)
            if width < best:
                best, best_order = width, order
        return best, best_order

    def lower(self) -> int:
        nodes = self.graph.nodes()
        if not nodes:
            return 0
        degree_floor = ceil(max(self.graph.degree(n) for n in nodes) / 2)
        diameter = 0
        for n in nodes:
            hops = self._hops(n)
            if len(hops) == len(nodes):
                diameter = max(diameter, *hops.values())
        distance_floor = ceil((len(nodes) - 1) / diameter) if diameter else 0
        return max(degree_floor, distance_floor)

    def exact(self) -> int:
        nodes = self.graph.nodes()
        if len(nodes) > 9:
            raise Invalid("the exact bandwidth tries every ordering; keep it under ten nodes")
        return min((self.span(self.graph, list(p)) for p in permutations(nodes)), default=0)

    def note(self) -> str:
        upper, _order = self.upper()
        lower = self.lower()
        meet = "the bounds meet" if upper == lower else f"a gap of {upper - lower}"
        return f"bandwidth at most {upper} from the level ordering, at least {lower}; {meet}"
