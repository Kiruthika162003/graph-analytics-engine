"""Greedy spanner: keep few edges, but never stretch any distance by more than t.

A spanning tree keeps the graph connected with the fewest edges, but it
can stretch distances badly: two nodes that were one edge apart may end
up on opposite ends of a long tree path. A t-spanner is the compromise.
It is a subgraph in which the distance between every pair of nodes is
at most t times their distance in the original, so it may drop many
edges but never lengthens any route by more than the stretch factor t.
Network designers want it for backbones that keep latency bounded with
far fewer links, and it is the structure behind approximate distance
oracles. The greedy construction is the classic one and it is exactly
Kruskal with a different test. Sort the edges by weight and consider
them lightest first; keep an edge only if, in the spanner built so far,
the distance between its endpoints exceeds t times the edge's weight,
because if the spanner already connects them within that bound the edge
adds nothing the guarantee needs. Every dropped edge is therefore
already spanned within stretch t, and since any path in the original is
a chain of edges each spanned within t, every path is spanned within t,
which is the whole proof. With t equal to one nothing can be dropped
except exact duplicates and the spanner is the graph; as t grows the
spanner thins toward a spanning tree, which is the limit as t goes to
infinity on a connected graph. The engine builds the spanner, verifies
the stretch on every pair by comparing all-pairs distances in the
spanner against the original, and reports the edge count kept against
the original and the largest stretch actually observed, because the
observed stretch is usually well under t and the gap between them is
slack the caller could have spent on dropping more edges.
"""

from __future__ import annotations

from mesh.dijkstra import Dijkstra
from mesh.errors import Invalid
from mesh.floydwarshall import FloydWarshall
from mesh.graph import Graph


class GreedySpanner:
    def __init__(self, graph: Graph, stretch: float) -> None:
        if graph.directed:
            raise Invalid("this spanner construction is for undirected graphs")
        if stretch < 1.0:
            raise Invalid("the stretch factor cannot be below one; distances cannot shrink")
        self.graph = graph
        self.stretch = stretch
        self.spanner = Graph()
        for n in graph.nodes():
            self.spanner.add_node(n)
        self._build()

    def _build(self) -> None:
        # Kruskal's order with the stretch test in place of the cycle test
        for u, v, w in sorted(self.graph.edges(), key=lambda e: (e[2], e[0], e[1])):
            run = Dijkstra(self.spanner, u)
            current = run.distance.get(v, float("inf"))
            if current > self.stretch * w:
                self.spanner.add_edge(u, v, w)

    def kept(self) -> int:
        return self.spanner.edge_count()

    def observed_stretch(self) -> float:
        # the largest ratio of spanner distance to original distance over all pairs
        original = FloydWarshall(self.graph)
        thinned = FloydWarshall(self.spanner)
        worst = 1.0
        nodes = self.graph.nodes()
        for i, a in enumerate(nodes):
            for b in nodes[i + 1 :]:
                d = original.dist[a][b]
                if d == float("inf") or d == 0:
                    continue
                worst = max(worst, thinned.dist[a][b] / d)
        return worst

    def stretch_holds(self) -> bool:
        return self.observed_stretch() <= self.stretch + 1e-9

    def note(self) -> str:
        return (
            f"kept {self.kept()} of {self.graph.edge_count()} edge(s) at stretch "
            f"{self.stretch}, worst observed {self.observed_stretch():.2f}; the gap "
            "under the bound is slack that could have dropped more edges"
        )
