"""Boruvka: every component grabs its cheapest outgoing edge, in parallel rounds.

Boruvka is the oldest minimum spanning tree algorithm and the one that
parallelizes most naturally. Instead of scanning edges one at a time like
Kruskal or growing a single tree like Prim, it works in rounds. In each
round every current component, starting with each node as its own
component, looks at all edges leaving it and picks the cheapest one that
reaches a different component. All of those picked edges are added at
once, merging components pairwise or in chains, and the round ends. Because
every component merges with at least one other in every round, the number
of components at least halves each time, so there are at most a logarithm
of rounds, each costing a scan of the edges, for a total of edges times log
nodes, matching Kruskal without the sort. The choice of the cheapest edge
out of each component is safe by the same cut property that justifies the
other two: a component and the rest of the graph form a cut, and the
lightest edge across it belongs to some minimum spanning tree. One hazard
needs care. When two components pick the same edge from opposite sides
that edge is added once, fine, but when edge weights tie, two components
could each pick a different edge of the same weight in a way that closes a
cycle among three or more components in one round. Breaking ties
consistently by a total order on edges, weight then endpoints, makes the
cheapest-edge choice unambiguous and the cycle impossible, which this
engine does. Union-find tracks the components and rejects an edge whose
endpoints already merged earlier in the same round. The builder returns the
chosen edges, the total, and the round count, and reports rounds against
the logarithmic bound, because a run that took the full bound is a graph
merging slowly, pairwise, while fewer rounds means chains of components
collapsed together.
"""

from __future__ import annotations

import math

from mesh.errors import Invalid
from mesh.graph import Graph
from mesh.unionfind import UnionFind


class Boruvka:
    def __init__(self, graph: Graph) -> None:
        if graph.directed:
            raise Invalid("a spanning tree is defined for an undirected graph")
        self.graph = graph
        self.chosen: list[tuple[str, str, float]] = []
        self.rounds = 0
        self._uf = UnionFind()
        for node in graph.nodes():
            self._uf.add(node)
        self._build()

    def _build(self) -> None:
        edges = self.graph.edges()
        while self._uf.group_count() > 1:
            # each component's cheapest outgoing edge, ties broken by endpoints
            cheapest: dict[str, tuple[float, str, str]] = {}
            for u, v, w in edges:
                ru, rv = self._uf.find(u), self._uf.find(v)
                if ru == rv:
                    continue
                key = (w, min(u, v), max(u, v))
                for root in (ru, rv):
                    if root not in cheapest or key < cheapest[root]:
                        cheapest[root] = key
            if not cheapest:
                break  # no edge leaves any component: the graph is disconnected
            self.rounds += 1
            for w, u, v in sorted(set(cheapest.values())):
                if self._uf.union(u, v):
                    self.chosen.append((u, v, w))

    def total_weight(self) -> float:
        return sum(w for _u, _v, w in self.chosen)

    def tree_count(self) -> int:
        return self._uf.group_count()

    def spans(self) -> bool:
        return self.graph.node_count() > 0 and self.tree_count() == 1

    def edges(self) -> list[tuple[str, str, float]]:
        return list(self.chosen)

    def round_bound(self) -> int:
        n = self.graph.node_count()
        return math.ceil(math.log2(n)) if n > 1 else 0

    def note(self) -> str:
        return (
            f"{self.rounds} round(s) against a bound of {self.round_bound()}, "
            f"total weight {self.total_weight()}; fewer rounds means chains of "
            "components collapsed together rather than merging pairwise"
        )
