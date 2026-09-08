"""Vertex separator: the actual nodes whose removal parts two others, not just how many.

Menger gives the count of node-disjoint routes between two nodes, which
equals the size of the smallest set of other nodes that separates them.
Often the count is not enough: an operator wants to know which routers
to harden, a biologist which proteins sit between two pathways, and
that is the separator itself. The engine recovers it from the same
node-splitting construction the count came from. Every intermediate
node becomes an in-half and an out-half joined by a unit edge, original
edges run from out-halves to in-halves with unbounded capacity so only
the internal unit edges can be saturated, and a maximum flow from the
source to the target saturates exactly a minimum set of those internal
edges. Reading the cut off the residual graph, the nodes the source can
still reach along unsaturated edges form the source side, and a node
whose in-half is reachable but whose out-half is not has its internal
edge cut: that node is in the separator. Because the original edges are
uncapacitated, the residual reachability cannot cross one of them
backward-only, so the crossing edges of the cut are internal edges and
nothing else, which is what makes the set of cut nodes a genuine
separator. The engine verifies it directly, removing the set and
checking the target is no longer reachable, and verifies minimality by
its size matching Menger's node-disjoint count. It refuses adjacent
endpoints, since no set of other nodes separates two nodes joined by an
edge, and reports the separator beside the count, because a separator
of size one is the single point of failure and a large one is a pair of
nodes that are hard to keep apart.
"""

from __future__ import annotations

from mesh.bfs import BFS
from mesh.edmondskarp import EdmondsKarp
from mesh.errors import Invalid, Missing
from mesh.graph import Graph
from mesh.menger import Menger


class VertexSeparator:
    def __init__(self, graph: Graph, a: str, b: str) -> None:
        if a == b:
            raise Invalid("a node cannot be separated from itself")
        for n in (a, b):
            if not graph.has_node(n):
                raise Missing(f"'{n}' is not in the graph")
        if graph.has_edge(a, b) or (not graph.directed and graph.has_edge(b, a)):
            raise Invalid(f"'{a}' and '{b}' are adjacent; no other nodes can separate them")
        self.graph = graph
        self.a = a
        self.b = b
        self.separator = self._extract()

    def _extract(self) -> set[str]:
        big = float(self.graph.node_count() + 1)
        g = Graph(directed=True)
        for n in self.graph.nodes():
            if n in (self.a, self.b):
                g.add_node(n)
            else:
                g.add_node(f"{n}\0in")
                g.add_node(f"{n}\0out")
                g.add_edge(f"{n}\0in", f"{n}\0out", 1.0)

        def tail(n: str) -> str:
            return n if n in (self.a, self.b) else f"{n}\0out"

        def head(n: str) -> str:
            return n if n in (self.a, self.b) else f"{n}\0in"

        for u, v, _w in self.graph.edges():
            g.add_edge(tail(u), head(v), big)  # only internal edges can saturate
            if not self.graph.directed:
                g.add_edge(tail(v), head(u), big)
        flow = EdmondsKarp(g, self.a, self.b)
        reach = {self.a}
        stack = [self.a]
        while stack:
            u = stack.pop()
            for v, cap in flow.residual[u].items():
                if cap > 0 and v not in reach:
                    reach.add(v)
                    stack.append(v)
        # a node whose in-half is reached but whose out-half is not was cut
        return {
            n for n in self.graph.nodes()
            if n not in (self.a, self.b) and f"{n}\0in" in reach and f"{n}\0out" not in reach
        }

    def separates(self) -> bool:
        trimmed = Graph(directed=self.graph.directed)
        for n in self.graph.nodes():
            if n not in self.separator:
                trimmed.add_node(n)
        for u, v, w in self.graph.edges():
            if u not in self.separator and v not in self.separator:
                trimmed.add_edge(u, v, w)
        return not BFS(trimmed, self.a).reached(self.b)

    def is_minimum(self) -> bool:
        return len(self.separator) == Menger(self.graph, self.a, self.b).node_disjoint

    def note(self) -> str:
        size = len(self.separator)
        shape = "a single point of failure" if size == 1 else \
            "nothing to cut, already apart" if size == 0 else "hard to keep apart"
        return (
            f"separator of {size} node(s) {sorted(self.separator)} parts '{self.a}' from "
            f"'{self.b}': {shape}; minimal by Menger: {self.is_minimum()}"
        )
