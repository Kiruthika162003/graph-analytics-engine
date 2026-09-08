"""Menger's theorem: how many separate routes join two nodes, edge-wise and node-wise.

Two nodes are well connected if there are many routes between them that
share nothing, so that no single failure can cut them off. Menger's
theorem makes that precise in two forms. The maximum number of
edge-disjoint paths between two nodes equals the minimum number of edges
whose removal separates them, and the maximum number of internally
node-disjoint paths equals the minimum number of intermediate nodes whose
removal separates them. Both are max-flow min-cut in disguise, which is
how the engine computes them. For edge-disjoint paths, give every edge
capacity one and run a maximum flow from one node to the other; each
unit of flow traces a path and no edge can carry two units, so the flow
value is the path count. For node-disjoint paths, split every
intermediate node into an in-half and an out-half joined by a single
edge of capacity one, route every original edge from an out-half to an
in-half, and run the same flow; now no intermediate node can be crossed
twice because its internal edge carries at most one unit. The
node-disjoint count is never more than the edge-disjoint count, since
node-disjoint paths are automatically edge-disjoint, and the gap between
them is a diagnostic: many edge-disjoint routes but few node-disjoint
ones means the routes fan out from a shared node, a hub that is the true
single point of failure even though the edges look redundant. The
counter returns both numbers for an undirected or directed graph, refuses
a pair of equal nodes, and reports them side by side, because the
edge-disjoint count is what the cabling suggests and the node-disjoint
count is what actually survives a router dying.
"""

from __future__ import annotations

from mesh.edmondskarp import EdmondsKarp
from mesh.errors import Invalid, Missing
from mesh.graph import Graph


class Menger:
    def __init__(self, graph: Graph, a: str, b: str) -> None:
        if a == b:
            raise Invalid("connectivity between a node and itself is not defined")
        for n in (a, b):
            if not graph.has_node(n):
                raise Missing(f"'{n}' is not in the graph")
        self.graph = graph
        self.a = a
        self.b = b
        self.edge_disjoint = self._edge_disjoint()
        self.node_disjoint = self._node_disjoint()

    def _unit_digraph(self) -> Graph:
        # every edge, in both directions if undirected, with capacity one
        g = Graph(directed=True)
        for n in self.graph.nodes():
            g.add_node(n)
        for u, v, _w in self.graph.edges():
            g.add_edge(u, v, 1.0)
            if not self.graph.directed:
                g.add_edge(v, u, 1.0)
        return g

    def _edge_disjoint(self) -> int:
        return int(EdmondsKarp(self._unit_digraph(), self.a, self.b).value)

    def _node_disjoint(self) -> int:
        # split each intermediate node into in and out halves joined by one unit
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
            g.add_edge(tail(u), head(v), 1.0)
            if not self.graph.directed:
                g.add_edge(tail(v), head(u), 1.0)
        return int(EdmondsKarp(g, self.a, self.b).value)

    def shared_hub_gap(self) -> int:
        return self.edge_disjoint - self.node_disjoint

    def note(self) -> str:
        return (
            f"{self.edge_disjoint} edge-disjoint and {self.node_disjoint} "
            f"node-disjoint route(s) between '{self.a}' and '{self.b}'; a gap of "
            f"{self.shared_hub_gap()} is routes fanning through a shared hub"
        )
