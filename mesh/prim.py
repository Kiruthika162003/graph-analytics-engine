"""Prim: grow a minimum spanning tree outward from one node, cheapest edge first.

Prim builds the same minimum spanning tree as Kruskal but grows it as a
single connected blob rather than assembling scattered edges. It starts
from one node and repeatedly adds the cheapest edge that leads from the tree
built so far to a node not yet in it, absorbing that node, until every
reachable node is inside. Where Kruskal sorts all edges once and uses
union-find to avoid cycles, Prim keeps a priority queue of the edges on the
frontier, the edges with exactly one endpoint in the tree, and pops the
lightest each step; a cycle is impossible by construction because it only
ever adds an edge to a node outside the tree. The safety is again the cut
property: at every step the tree so far is one side of a cut and the rest is
the other, and the lightest edge crossing that cut is safe to add, which is
exactly the edge Prim pops. Prim tends to win on dense graphs where sorting
all edges is expensive, and Kruskal on sparse ones, but on a connected
graph they must produce trees of identical total weight, which is a claim
this engine checks against Kruskal directly. On a disconnected graph Prim
from one node reaches only that node's component, so it spans a single
component and reports that it could not reach the rest, rather than
pretending a spanning tree exists. The builder returns the chosen edges and
their total, tells whether it spanned the whole graph, and reports the
reached-node count against the graph size and the total weight, the number
that must match Kruskal's on a connected graph.
"""

from __future__ import annotations

import heapq

from mesh.errors import Invalid, Missing
from mesh.graph import Graph


class Prim:
    def __init__(self, graph: Graph, start: str | None = None) -> None:
        if graph.directed:
            raise Invalid("a spanning tree is defined for an undirected graph")
        if graph.node_count() == 0:
            raise Invalid("an empty graph has no spanning tree")
        if start is None:
            start = graph.nodes()[0]
        if not graph.has_node(start):
            raise Missing(f"start '{start}' is not in the graph")
        self.graph = graph
        self.start = start
        self.chosen: list[tuple[str, str, float]] = []
        self._build()

    def _build(self) -> None:
        in_tree: set[str] = {self.start}
        # the frontier heap holds (weight, from-in-tree, to-outside) edges
        frontier: list[tuple[float, str, str]] = []
        for nbr, w in self.graph.neighbors(self.start).items():
            heapq.heappush(frontier, (w, self.start, nbr))
        while frontier:
            weight, src, dst = heapq.heappop(frontier)
            if dst in in_tree:
                continue  # both ends now inside, this frontier edge is stale
            in_tree.add(dst)
            self.chosen.append((src, dst, weight))
            for nbr, w in self.graph.neighbors(dst).items():
                if nbr not in in_tree:
                    heapq.heappush(frontier, (w, dst, nbr))

    def total_weight(self) -> float:
        return sum(w for _u, _v, w in self.chosen)

    def reached_count(self) -> int:
        return len(self.chosen) + 1

    def spans(self) -> bool:
        return self.reached_count() == self.graph.node_count()

    def edges(self) -> list[tuple[str, str, float]]:
        return list(self.chosen)

    def note(self) -> str:
        return (
            f"grew a tree reaching {self.reached_count()} of "
            f"{self.graph.node_count()} node(s), total weight "
            f"{self.total_weight()}; on a connected graph this equals Kruskal's"
        )
