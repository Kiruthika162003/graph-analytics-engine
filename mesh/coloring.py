"""Graph coloring: assign colors so no edge joins two alike, using as few as you can.

A proper coloring gives every node a color such that no two adjacent
nodes share one. It is exam scheduling, where two exams with a common
student cannot share a slot; register allocation, where two variables live
at the same time cannot share a register; frequency assignment, where two
towers in range cannot share a channel. The smallest number of colors that
works is the chromatic number, and finding it exactly is NP-hard, so what
an engine can do efficiently is find a proper coloring that is good, and
be honest that it may not be optimal. The greedy method walks the nodes in
some order and gives each the smallest color not already used by a
neighbor. Whatever the order, greedy never needs more colors than the
maximum degree plus one, because a node has at most max-degree neighbors
and so at most that many colors are blocked, leaving one free. The order
matters a great deal for how close to optimal greedy lands. Welsh-Powell
orders by degree descending, coloring the most constrained nodes while
colors are plentiful, and tends to beat an arbitrary order. DSatur is
adaptive: it repeatedly picks the uncolored node whose neighbors already
use the most distinct colors, its saturation, breaking ties by degree, so
the nodes with the fewest options are handled first while options still
exist. DSatur is exact on bipartite graphs and usually stronger than
Welsh-Powell in practice, at the cost of recomputing saturation as it
goes. The colorer runs greedy under any of the three orders, verifies the
result is proper, and reports the color count against the max-degree-plus-
one bound and against the largest clique it can find cheaply, the
triangle-level lower bound, so the reported count sits between a bound it
cannot beat and one it is guaranteed not to exceed.
"""

from __future__ import annotations

from mesh.errors import Invalid
from mesh.graph import Graph


class Coloring:
    def __init__(self, graph: Graph, strategy: str = "dsatur") -> None:
        if graph.directed:
            raise Invalid("coloring is defined on an undirected graph")
        if strategy not in ("arbitrary", "welsh_powell", "dsatur"):
            raise Invalid(f"unknown strategy '{strategy}'")
        self.graph = graph
        self.strategy = strategy
        self._neighbors = {n: set(graph.neighbors(n)) for n in graph.nodes()}
        self.color: dict[str, int] = {}
        self._run()

    def _smallest_free(self, node: str) -> int:
        used = {self.color[m] for m in self._neighbors[node] if m in self.color}
        c = 0
        while c in used:
            c += 1
        return c

    def _run(self) -> None:
        if self.strategy == "dsatur":
            self._dsatur()
            return
        order = self.graph.nodes()
        if self.strategy == "welsh_powell":
            order = sorted(order, key=lambda n: (-len(self._neighbors[n]), n))
        for node in order:
            self.color[node] = self._smallest_free(node)

    def _dsatur(self) -> None:
        # pick the uncolored node with the most distinct neighbor colors,
        # so the nodes with the fewest options go first while options remain
        uncolored = set(self.graph.nodes())
        while uncolored:
            def saturation(n: str) -> int:
                return len({self.color[m] for m in self._neighbors[n] if m in self.color})

            node = max(uncolored, key=lambda n: (saturation(n), len(self._neighbors[n]), n))
            self.color[node] = self._smallest_free(node)
            uncolored.discard(node)

    def is_proper(self) -> bool:
        return all(self.color[u] != self.color[v] for u, v, _w in self.graph.edges())

    def color_count(self) -> int:
        return len(set(self.color.values())) if self.color else 0

    def upper_bound(self) -> int:
        degrees = [len(s) for s in self._neighbors.values()]
        return max(degrees, default=0) + 1

    def lower_bound(self) -> int:
        # the triangle-level clique bound: 3 if any triangle, else 2 if any edge
        for u in self.graph.nodes():
            for v in self._neighbors[u]:
                if self._neighbors[u] & self._neighbors[v]:
                    return 3
        return 2 if self.graph.edge_count() else 1

    def note(self) -> str:
        return (
            f"{self.strategy} used {self.color_count()} color(s), between a lower "
            f"bound of {self.lower_bound()} and max-degree-plus-one of "
            f"{self.upper_bound()}; the count may not be optimal, only proper"
        )
