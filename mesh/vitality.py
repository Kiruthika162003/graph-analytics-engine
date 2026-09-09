"""Closeness vitality: how much the world's distances grow when one node or edge is gone.

Betweenness asks how many shortest paths pass through a node; vitality
asks what happens to everyone's distances when the node is taken away.
The closeness vitality of a node is the Wiener index of the graph with
the node removed subtracted from the Wiener index of the whole graph,
where the removed node's own pairs no longer count. A node whose
removal leaves every remaining distance as it was has vitality equal
to the sum of its own distances, which is the largest it can be; a
node that sat on many shortest paths forces detours, and the detours
lengthen the remaining distances and pull the score down, below zero
when they are long enough, which is what a cycle node shows; a node
whose removal disconnects the graph scores infinity, because some
pair loses its route entirely. The same reading for an edge is
the growth of the Wiener index when the edge is cut, which is zero for
an edge no shortest path needs and infinity for a bridge. The closed
forms are exact on the shapes chemists first computed: a leaf of a
star on n nodes scores 2n minus 3, every node of a complete graph
scores n minus 1, and an end of a path on n nodes scores n(n minus 1)
over 2, the sum of its distances, since removing it shifts nothing
else. The engine computes all node vitalities and all edge vitalities
by rebuilding the graph without the item and reusing the Wiener
module, reports the highest-scoring node as the most vital, and
refuses a directed graph.
"""

from __future__ import annotations

from math import inf

from mesh.errors import Invalid
from mesh.graph import Graph
from mesh.wienerindex import WienerIndex


class Vitality:
    def __init__(self, graph: Graph) -> None:
        if graph.directed:
            raise Invalid("vitality is read on an undirected graph")
        self.graph = graph
        self.whole = WienerIndex(graph).wiener()

    def _without_node(self, node: str) -> Graph:
        g = Graph()
        for n in self.graph.nodes():
            if n != node:
                g.add_node(n)
        for u, v, w in self.graph.edges():
            if node not in (u, v):
                g.add_edge(u, v, w)
        return g

    def _without_edge(self, a: str, b: str) -> Graph:
        g = Graph()
        for n in self.graph.nodes():
            g.add_node(n)
        for u, v, w in self.graph.edges():
            if {u, v} != {a, b}:
                g.add_edge(u, v, w)
        return g

    def node(self, node: str) -> float:
        if node not in self.graph.nodes():
            raise Invalid(f"'{node}' is not a node of the graph")
        if self.whole == inf:
            return inf
        rest = WienerIndex(self._without_node(node)).wiener()
        return inf if rest == inf else self.whole - rest

    def edge(self, a: str, b: str) -> float:
        if not self.graph.has_edge(a, b):
            raise Invalid(f"no edge {a}-{b} to cut")
        if self.whole == inf:
            return inf
        rest = WienerIndex(self._without_edge(a, b)).wiener()
        return inf if rest == inf else rest - self.whole

    def all_nodes(self) -> dict[str, float]:
        return {n: self.node(n) for n in self.graph.nodes()}

    def all_edges(self) -> dict[tuple[str, str], float]:
        return {(u, v): self.edge(u, v) for u, v, _w in self.graph.edges()}

    def most_vital(self) -> str | None:
        scores = self.all_nodes()
        if not scores:
            return None
        return max(sorted(scores), key=lambda n: scores[n])

    def own_distance_sum(self, node: str) -> float:
        table = WienerIndex(self.graph).table[node]
        return sum(d for m, d in table.items() if m != node)

    def note(self) -> str:
        top = self.most_vital()
        if top is None:
            return "no nodes, nothing vital"
        score = self.node(top)
        cut = "disconnects the graph" if score == inf else f"adds {score:g} to the distance sum"
        return f"most vital node {top}: removing it {cut}"
