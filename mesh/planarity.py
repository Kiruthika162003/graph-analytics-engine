"""Planarity obstructions: two checks that prove a graph cannot be drawn flat.

A planar graph can be drawn on a page with no two edges crossing, and
whether a graph is planar matters for circuit layout, map drawing, and
for the many algorithms that run faster on planar inputs. Deciding it
exactly takes a linear-time algorithm of real intricacy, and this
engine does not pretend to have one. What it has are two obstructions,
each of which proves non-planarity when found, and an honest third
verdict when neither is. The first is Euler's bound: a planar graph
with at least three nodes has at most three times the node count minus
six edges, and a triangle-free planar graph at most twice the node
count minus four, so a graph over either bound cannot be drawn flat,
no search needed. The second is Kuratowski's pair: the complete graph
on five nodes and the complete bipartite graph on three and three are
the two minimal non-planar graphs, and any graph containing either as a
subgraph is non-planar. The engine looks for them with the subgraph
matcher. Kuratowski's full theorem says a graph is non-planar exactly
when it contains a subdivision of one of the two, edges replaced by
paths, and subdivisions are not subgraphs, so a graph can be non-planar
while containing neither directly; the engine's third verdict, no
obstruction found, therefore means exactly that and not planar, and the
docstring and the note both say so rather than rounding it up to a
claim of planarity. The checker returns the verdict and which
obstruction it found, and reports the edge count against the Euler
bound, because a graph sitting right at the bound is a maximal planar
graph if it is planar at all, and one far below it has room to spare.
"""

from __future__ import annotations

from itertools import combinations

from mesh.errors import Invalid
from mesh.graph import Graph
from mesh.subgraphmatch import SubgraphMatch
from mesh.triangles import Triangles


def _k5() -> Graph:
    g = Graph()
    nodes = [f"k{i}" for i in range(5)]
    for n in nodes:
        g.add_node(n)
    for a, b in combinations(nodes, 2):
        g.add_edge(a, b)
    return g


def _k33() -> Graph:
    g = Graph()
    left = [f"l{i}" for i in range(3)]
    right = [f"r{i}" for i in range(3)]
    for n in left + right:
        g.add_node(n)
    for a in left:
        for b in right:
            g.add_edge(a, b)
    return g


class PlanarityCheck:
    def __init__(self, graph: Graph) -> None:
        if graph.directed:
            raise Invalid("planarity is a property of the undirected shape")
        self.graph = graph
        self.obstruction = ""
        self.verdict = self._decide()

    def euler_bound(self) -> int:
        n = self.graph.node_count()
        if n < 3:
            return self.graph.edge_count()
        triangle_free = Triangles(self.graph).total == 0
        return 2 * n - 4 if triangle_free else 3 * n - 6

    def _decide(self) -> str:
        if self.graph.node_count() >= 3 and self.graph.edge_count() > self.euler_bound():
            self.obstruction = "Euler bound"
            return "non-planar"
        for name, pattern in (("K5", _k5()), ("K3,3", _k33())):
            fits = pattern.node_count() <= self.graph.node_count()
            if fits and SubgraphMatch(self.graph, pattern).occurs():
                self.obstruction = f"{name} subgraph"
                return "non-planar"
        self.obstruction = "none found"
        return "no obstruction found"

    def is_proven_non_planar(self) -> bool:
        return self.verdict == "non-planar"

    def note(self) -> str:
        return (
            f"{self.verdict} ({self.obstruction}); {self.graph.edge_count()} edge(s) "
            f"against an Euler bound of {self.euler_bound()}. No obstruction found is "
            "not a proof of planarity: subdivisions are not subgraphs"
        )
