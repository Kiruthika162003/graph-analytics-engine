"""Kruskal: build a minimum spanning tree by taking cheap edges that do not loop.

A minimum spanning tree of a connected undirected graph is a subset of
edges that touches every node, forms no cycle, and has the least total
weight among all such subsets: the cheapest way to wire every node into one
piece. Kruskal builds it greedily by edge. It sorts every edge by weight
and considers them cheapest first, taking an edge into the tree if its two
endpoints are not already connected and discarding it if they are, because
an edge between two already-connected nodes would close a cycle and a tree
has none. The connectivity check is exactly union-find: each accepted edge
unions its endpoints, and an edge is rejected precisely when find already
puts its endpoints in the same group. The greedy choice is safe by the cut
property, that the lightest edge crossing any partition of the nodes into
two sides is safe to include in some minimum spanning tree, and taking
edges in weight order always adds the lightest edge that joins two
currently-separate pieces. The tree is complete when it has one fewer edge
than there are nodes; if the edges run out before that, the graph was
disconnected and has no spanning tree, only a spanning forest, and the
builder reports that honestly rather than returning a partial tree that
silently omits a component. On a disconnected graph it returns the minimum
spanning forest and says how many trees it holds. The builder returns the
chosen edges and their total weight, tells whether the graph spanned into
one tree, and reports the tree count and total weight, the number a network
designer compares against Prim to confirm both greedy routes reach the same
minimum.
"""

from __future__ import annotations

from mesh.errors import Invalid
from mesh.graph import Graph
from mesh.unionfind import UnionFind


class Kruskal:
    def __init__(self, graph: Graph) -> None:
        if graph.directed:
            raise Invalid("a spanning tree is defined for an undirected graph")
        self.graph = graph
        self.chosen: list[tuple[str, str, float]] = []
        self._uf = UnionFind()
        for node in graph.nodes():
            self._uf.add(node)
        self._build()

    def _build(self) -> None:
        # consider edges cheapest first; take one only if it joins two pieces
        edges = sorted(self.graph.edges(), key=lambda e: e[2])
        for u, v, w in edges:
            if self._uf.union(u, v):
                self.chosen.append((u, v, w))

    def total_weight(self) -> float:
        return sum(w for _u, _v, w in self.chosen)

    def tree_count(self) -> int:
        # a spanning forest has one tree per connected component
        return self._uf.group_count()

    def spans(self) -> bool:
        return self.graph.node_count() > 0 and self.tree_count() == 1

    def edges(self) -> list[tuple[str, str, float]]:
        return list(self.chosen)

    def note(self) -> str:
        if self.spans():
            return (
                f"minimum spanning tree of {len(self.chosen)} edge(s), total "
                f"weight {self.total_weight()}; must equal Prim's total"
            )
        return (
            f"graph is disconnected: minimum spanning forest of "
            f"{self.tree_count()} tree(s), total weight {self.total_weight()}"
        )
