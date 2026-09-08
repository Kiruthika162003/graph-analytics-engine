"""Vertex cover: guard every edge with the fewest nodes, exactly where that is possible.

A vertex cover is a set of nodes that touches every edge, so that placing
a guard at each chosen node watches every connection in the graph. The
minimum vertex cover is the smallest such set, and finding it on a general
graph is NP-hard, one of the original hard problems. Two things are still
within reach. On a bipartite graph the problem is easy, by Konig's
theorem: the size of the minimum vertex cover equals the size of the
maximum matching, and the cover itself can be read off a maximum matching
by an alternating-path argument. Starting from the unmatched left nodes,
walk alternating paths, unmatched edges leftward to rightward and matched
edges back, and mark every node reached; the cover is the left nodes not
marked together with the right nodes that are marked. That set touches
every edge and has exactly one node per matched edge. On a general graph
the honest answer is an approximation with a proof: take any maximal
matching, a set of disjoint edges no further edge can join, and put both
endpoints of every matched edge in the cover. Every edge touches a matched
edge's endpoint, or it could have been added to the matching, so the set
covers; and any cover must include at least one endpoint of each matched
edge since they are disjoint, so the minimum is at least the matching
size, and this cover is at most twice it. A factor of two, guaranteed, is
the best simple method known, and the engine reports which regime it is
in rather than presenting the approximation as exact. The solver detects
bipartiteness, computes the exact cover through Hopcroft-Karp when it can
and the two-approximation otherwise, verifies that the returned set covers
every edge, and reports the cover size against the matching lower bound,
because that ratio is either exactly one, when the answer is exact, or
the slack the approximation left on the table.
"""

from __future__ import annotations

from collections import deque

from mesh.bipartite import Bipartite
from mesh.errors import Invalid
from mesh.graph import Graph
from mesh.hopcroftkarp import HopcroftKarp


class VertexCover:
    def __init__(self, graph: Graph) -> None:
        if graph.directed:
            raise Invalid("vertex cover is defined on an undirected graph")
        self.graph = graph
        self._bip = Bipartite(graph)
        self.exact = self._bip.is_bipartite
        self.lower_bound = 0
        self.cover: set[str] = self._konig() if self.exact else self._two_approx()

    def _konig(self) -> set[str]:
        left, right = self._bip.sides()
        hk = HopcroftKarp(sorted(left), sorted(right))
        for u, v, _w in self.graph.edges():
            if u in left:
                hk.add_edge(u, v)
            else:
                hk.add_edge(v, u)
        self.lower_bound = hk.solve()
        matched = hk.matching()
        matched_right = {v: u for u, v in matched.items()}
        # alternating BFS from every unmatched left node
        marked: set[str] = set()
        queue: deque[str] = deque(u for u in left if u not in matched)
        marked.update(queue)
        while queue:
            u = queue.popleft()
            for v in self.graph.neighbors(u):
                if matched.get(u) == v or v in marked:
                    continue  # only unmatched edges lead left to right
                marked.add(v)
                w = matched_right.get(v)
                if w is not None and w not in marked:
                    marked.add(w)  # matched edges lead back to the left
                    queue.append(w)
        return {u for u in left if u not in marked} | {v for v in right if v in marked}

    def _two_approx(self) -> set[str]:
        # both endpoints of a maximal matching: at most twice the optimum
        used: set[str] = set()
        cover: set[str] = set()
        for u, v, _w in self.graph.edges():
            if u not in used and v not in used:
                used.update((u, v))
                cover.update((u, v))
                self.lower_bound += 1
        return cover

    def covers_every_edge(self) -> bool:
        return all(u in self.cover or v in self.cover for u, v, _w in self.graph.edges())

    def note(self) -> str:
        regime = "exact by Konig" if self.exact else "within a factor of two"
        return (
            f"cover of {len(self.cover)} node(s), {regime}, matching bound "
            f"{self.lower_bound}; the gap above the bound is the slack the "
            "approximation left, zero when exact"
        )
