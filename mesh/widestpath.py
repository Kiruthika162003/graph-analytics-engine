"""Widest path: the route whose narrowest link is as wide as possible.

Shortest path minimizes a sum; widest path maximizes a minimum. Each
edge carries a capacity, the bandwidth of a link or the clearance of a
bridge, a route's width is the smallest capacity along it, and the
widest path is the route whose bottleneck is largest. It is what a
video call wants when it picks a path through the internet and what a
haulier wants when the tallest truck must fit under every overpass on
the way. Dijkstra adapts with two changes: the value of a node is the
best bottleneck found so far instead of the best sum, a candidate
through an edge is the minimum of the current value and the edge's
capacity instead of the sum, and the heap pops the largest value first
instead of the smallest. The correctness argument survives because
taking a minimum along a path can only lower the width, the mirror of
sums only raising the length, so the node with the widest tentative
route can never later be improved by a route through a narrower one.
There is a second way to the same answer that makes a good check: the
widest path between any two nodes lies in the maximum spanning tree,
because the cut property for maximum weight says the widest edge across
any cut is in it, so a Kruskal run on negated weights yields a tree
whose unique path between two nodes is a widest path. The engine
computes the widest path by the adapted Dijkstra, verifies it against
the maximum-spanning-tree path, refuses a negative capacity, and reports
the width against the largest edge capacity leaving the source, because
a width equal to that first edge means the bottleneck is at the very
start and a width well below it means the route narrows somewhere in
the middle where a single upgrade would widen the whole thing.
"""

from __future__ import annotations

import heapq
import math

from mesh.errors import Invalid, Missing, Unreachable
from mesh.graph import Graph
from mesh.kruskal import Kruskal


class WidestPath:
    def __init__(self, graph: Graph, source: str) -> None:
        if not graph.has_node(source):
            raise Missing(f"source '{source}' is not in the graph")
        for u, v, w in graph.edges():
            if w < 0:
                raise Invalid(f"edge {u}->{v} has negative capacity {w}")
        self.graph = graph
        self.source = source
        self.width: dict[str, float] = {source: math.inf}
        self.parent: dict[str, str | None] = {source: None}
        self._run()

    def _run(self) -> None:
        # a max-heap on width, stored negated; widest tentative node settles first
        heap: list[tuple[float, str]] = [(-math.inf, self.source)]
        settled: set[str] = set()
        while heap:
            neg, node = heapq.heappop(heap)
            if node in settled:
                continue
            settled.add(node)
            here = -neg
            for nbr, cap in self.graph.neighbors(node).items():
                candidate = min(here, cap)  # the bottleneck can only narrow
                if candidate > self.width.get(nbr, -math.inf):
                    self.width[nbr] = candidate
                    self.parent[nbr] = node
                    heapq.heappush(heap, (-candidate, nbr))

    def width_to(self, node: str) -> float:
        if node not in self.width:
            raise Unreachable(f"'{node}' is unreachable from '{self.source}'")
        return self.width[node]

    def path_to(self, node: str) -> list[str]:
        if node not in self.width:
            raise Unreachable(f"'{node}' is unreachable from '{self.source}'")
        path: list[str] = []
        current: str | None = node
        while current is not None:
            path.append(current)
            current = self.parent[current]
        path.reverse()
        return path

    def maximum_spanning_tree_width(self, node: str) -> float:
        # the widest path lives in the maximum spanning tree: check by that route
        if self.graph.directed:
            raise Invalid("the spanning-tree check needs an undirected graph")
        negated = Graph()
        for n in self.graph.nodes():
            negated.add_node(n)
        for u, v, w in self.graph.edges():
            negated.add_edge(u, v, -w)
        tree = Graph()
        for n in self.graph.nodes():
            tree.add_node(n)
        for u, v, w in Kruskal(negated).edges():
            tree.add_edge(u, v, -w)
        return WidestPath(tree, self.source).width_to(node)

    def first_hop_ceiling(self) -> float:
        return max(self.graph.neighbors(self.source).values(), default=0.0)

    def note(self, node: str) -> str:
        width = self.width_to(node)
        where = "at the very first hop" if width == self.first_hop_ceiling() else \
            "somewhere in the middle, where one upgrade widens the route"
        return f"widest route to '{node}' carries {width}; the bottleneck is {where}"
