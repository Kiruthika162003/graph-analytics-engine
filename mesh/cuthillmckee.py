"""Reverse Cuthill-McKee: number the nodes so every edge joins near neighbors.

The bandwidth of a node ordering is the largest gap, in position, across
any edge: number the nodes one to n and ask how far apart the two ends of
the worst edge sit. A small bandwidth means the adjacency matrix has all
its nonzeros in a narrow band around the diagonal, which is what a
banded linear solver, a cache-friendly traversal, or a sparse
factorization wants, since fill-in during elimination stays inside the
band. The Cuthill-McKee ordering shrinks it with a breadth-first search
that is careful about two choices. It starts from a node of low degree,
ideally a peripheral one, because starting in the middle of the graph
spreads the numbering in every direction at once; and it visits each
node's unvisited neighbors in increasing order of degree, so that the
nodes with the most edges still to place get low numbers close to their
already-placed neighbors. The reverse ordering, reading the result
backward, is almost always at least as good and often better for the
elimination fill-in, a fact George noticed empirically and the reason
the reversed form is the one in use. The engine picks the start by the
double sweep from a low-degree node, runs the degree-ordered BFS over
every component, reverses, and measures the bandwidth before and after
so the improvement is a number rather than a claim; on a path given in
a scrambled order the bandwidth drops from near n to one, and on a
dense graph it barely moves because no ordering can help a matrix that
is mostly full. The orderer returns the permutation, the old and new
bandwidth, and reports the ratio, because a ratio near one is a graph
whose structure gave the ordering nothing to exploit.
"""

from __future__ import annotations

from collections import deque

from mesh.bfs import BFS
from mesh.errors import Invalid
from mesh.graph import Graph


class ReverseCuthillMcKee:
    def __init__(self, graph: Graph) -> None:
        if graph.directed:
            raise Invalid("bandwidth ordering here is for a symmetric, undirected graph")
        if graph.node_count() == 0:
            raise Invalid("an empty graph has nothing to order")
        self.graph = graph
        self.original = graph.nodes()
        self.order = self._order()
        self.before = self.bandwidth(self.original)
        self.after = self.bandwidth(self.order)

    def bandwidth(self, order: list[str]) -> int:
        pos = {n: i for i, n in enumerate(order)}
        return max((abs(pos[u] - pos[v]) for u, v, _w in self.graph.edges()), default=0)

    def _peripheral_start(self, candidates: list[str]) -> str:
        # a low-degree node, then the far end of a sweep from it
        low = min(candidates, key=lambda n: (self.graph.degree(n), n))
        sweep = BFS(self.graph, low)
        return max(sweep.distance, key=lambda n: (sweep.distance[n], -self.graph.degree(n), n))

    def _order(self) -> list[str]:
        placed: list[str] = []
        seen: set[str] = set()
        while len(placed) < self.graph.node_count():
            remaining = [n for n in self.graph.nodes() if n not in seen]
            start = self._peripheral_start(remaining)
            seen.add(start)
            queue: deque[str] = deque([start])
            while queue:
                node = queue.popleft()
                placed.append(node)
                # unvisited neighbors by increasing degree, ties by name
                nbrs = sorted(
                    (m for m in self.graph.neighbors(node) if m not in seen),
                    key=lambda m: (self.graph.degree(m), m),
                )
                for m in nbrs:
                    seen.add(m)
                    queue.append(m)
        placed.reverse()
        return placed

    def improvement(self) -> float:
        return self.after / self.before if self.before else 1.0

    def note(self) -> str:
        return (
            f"bandwidth {self.before} to {self.after} (ratio {self.improvement():.2f}); a "
            "ratio near one is a graph whose structure gave the ordering nothing to use"
        )
