"""Triangles and clustering: do a node's friends know each other?

A triangle is three nodes all connected to one another, and the count of
triangles in a graph is the most basic measure of how clustered it is. In
a social network, a triangle is two friends of yours who are also friends
with each other, and networks of people are full of them where a random
graph of the same size and density would have almost none, so the
triangle count separates real social structure from noise. The local
clustering coefficient of a node makes this per-node: among all pairs of
its neighbors, the fraction that are themselves connected, which is the
triangles through the node divided by the pairs of neighbors it has. A
node whose neighbors all know each other scores one, a node at the center
of a star whose neighbors are strangers scores zero. Averaging over nodes
gives the graph's average clustering, and the global transitivity is the
ratio of three times the triangle count to the number of connected
triples, the paths of length two, which weights high-degree nodes more
than the average does. Counting triangles naively by checking every
triple of nodes is cubic. The engine counts them per node by intersecting
neighbor sets, and to avoid counting each triangle once from each corner it
counts a triangle only from its lowest-named corner, ordering the neighbor
scan so each triangle is seen exactly once. The counter returns the total
triangle count, each node's triangle count and clustering coefficient, the
average clustering, and the transitivity, refusing a directed graph since
a triangle there has orientation questions this measure does not answer.
It reports the average clustering against the density, because clustering
far above density is the signature of real community structure rather
than random connection.
"""

from __future__ import annotations

from mesh.errors import Invalid, Missing
from mesh.graph import Graph


class Triangles:
    def __init__(self, graph: Graph) -> None:
        if graph.directed:
            raise Invalid("triangle counting here is for undirected graphs")
        self.graph = graph
        self.per_node: dict[str, int] = dict.fromkeys(graph.nodes(), 0)
        self._neighbors = {n: set(graph.neighbors(n)) for n in graph.nodes()}
        self.total = self._count()

    def _count(self) -> int:
        total = 0
        for u in self.graph.nodes():
            for v in self._neighbors[u]:
                if v <= u:
                    continue  # count each triangle from its lowest corner only
                common = self._neighbors[u] & self._neighbors[v]
                for w in common:
                    if w > v:
                        total += 1
                        self.per_node[u] += 1
                        self.per_node[v] += 1
                        self.per_node[w] += 1
        return total

    def clustering(self, node: str) -> float:
        if node not in self._neighbors:
            raise Missing(f"node '{node}' is not in the graph")
        degree = len(self._neighbors[node])
        if degree < 2:
            return 0.0  # no pair of neighbors to be connected or not
        pairs = degree * (degree - 1) / 2
        return self.per_node[node] / pairs

    def average_clustering(self) -> float:
        nodes = self.graph.nodes()
        if not nodes:
            return 0.0
        return sum(self.clustering(n) for n in nodes) / len(nodes)

    def transitivity(self) -> float:
        triples = sum(
            d * (d - 1) / 2 for d in (len(s) for s in self._neighbors.values())
        )
        return 3 * self.total / triples if triples else 0.0

    def density(self) -> float:
        n = self.graph.node_count()
        if n < 2:
            return 0.0
        return self.graph.edge_count() / (n * (n - 1) / 2)

    def note(self) -> str:
        return (
            f"{self.total} triangle(s), average clustering "
            f"{self.average_clustering():.3f} against density {self.density():.3f}; "
            "clustering far above density is real community structure, not chance"
        )
