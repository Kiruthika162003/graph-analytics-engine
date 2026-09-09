"""Graphlet degree signature: how many small shapes each node touches, and in what role.

Degree counts the edges at a node. Graphlet degrees, introduced by
Przulj, count the small connected induced subgraphs a node touches,
split by the position the node occupies in each, so two nodes with
the same degree can still look different: one sits at the end of
paths, the other in the middle of triangles. The engine counts the
graphlets on two and three nodes, which give four positions, called
orbits. Orbit 0 is the edge, so it is the degree. Orbit 1 is an end
of an induced path on three nodes, orbit 2 its middle, and orbit 3 a
corner of a triangle. The counts obey identities that check the
arithmetic: the total over nodes of orbit 2 counts each induced path
once and orbit 1 counts it twice, orbit 3 totals three times the
triangle count, and the sum of orbits 2 and 3 at a node is its degree
choose 2, since every pair of neighbors is either joined, a triangle,
or not, a path through the node. The signature is the vector of four
orbit counts, the graphlet distance between two nodes is the sum over
orbits of the absolute difference of log counts scaled by the orbit's
weight, and the module ranks the nodes most unlike a chosen node,
which is how a signature finds a node's structural peers across a
network. A directed graph is refused since the shapes here are
undirected.
"""

from __future__ import annotations

from itertools import combinations
from math import log

from mesh.errors import Invalid
from mesh.graph import Graph


class Graphlets:
    def __init__(self, graph: Graph) -> None:
        if graph.directed:
            raise Invalid("graphlet orbits here are undirected shapes")
        self.graph = graph
        self.nodes = graph.nodes()
        self.signature: dict[str, list[int]] = {n: [0, 0, 0, 0] for n in self.nodes}
        self.paths = 0
        self.triangles = 0
        self._count()

    def _count(self) -> None:
        for node in self.nodes:
            self.signature[node][0] = self.graph.degree(node)
        for node in self.nodes:
            nbrs = sorted(self.graph.neighbors(node))
            for a, b in combinations(nbrs, 2):
                if self.graph.has_edge(a, b):
                    # a triangle is seen from each of its three corners; count once
                    if node < a and node < b:
                        self.triangles += 1
                        for corner in (node, a, b):
                            self.signature[corner][3] += 1
                else:
                    self.paths += 1
                    self.signature[node][2] += 1
                    self.signature[a][1] += 1
                    self.signature[b][1] += 1

    def identities_hold(self) -> bool:
        ends = sum(sig[1] for sig in self.signature.values())
        middles = sum(sig[2] for sig in self.signature.values())
        corners = sum(sig[3] for sig in self.signature.values())
        pairs_ok = all(
            sig[2] + sig[3] == sig[0] * (sig[0] - 1) // 2 for sig in self.signature.values()
        )
        totals_ok = ends == 2 * self.paths and middles == self.paths
        return totals_ok and corners == 3 * self.triangles and pairs_ok

    @staticmethod
    def distance(a: list[int], b: list[int]) -> float:
        # orbit weights fall with how many other orbits each depends on
        weights = [1.0, 0.5, 0.5, 1.0]
        total = 0.0
        for w, x, y in zip(weights, a, b, strict=True):
            total += w * abs(log(x + 1) - log(y + 1)) / log(max(x, y) + 2)
        return total / sum(weights)

    def peers(self, node: str) -> list[tuple[str, float]]:
        if node not in self.signature:
            raise Invalid(f"'{node}' is not a node of the graph")
        me = self.signature[node]
        others = [(n, self.distance(me, sig)) for n, sig in self.signature.items() if n != node]
        return sorted(others, key=lambda p: (p[1], p[0]))

    def note(self, node: str) -> str:
        sig = self.signature[node]
        return (
            f"{node}: degree {sig[0]}, path end {sig[1]} time(s), path middle {sig[2]}, "
            f"triangle corner {sig[3]}; {self.paths} induced path(s) and "
            f"{self.triangles} triangle(s) overall"
        )
