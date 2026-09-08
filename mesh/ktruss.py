"""K-truss: keep only the edges that sit in enough triangles, then keep going.

The k-core keeps nodes with enough neighbors; the k-truss keeps edges
with enough support, where an edge's support is the number of triangles
it belongs to. The k-truss of a graph is the largest subgraph in which
every edge lies in at least k minus two triangles, so the 3-truss drops
edges in no triangle, the 4-truss keeps only edges in two or more, and
so on until nothing survives. It is a stricter, more cohesive notion
than the core, because a hub with many mutually unacquainted neighbors
has high degree but its edges have no support, so it sits in a high core
and a low truss: the truss finds groups where the connections themselves
are reinforced, the shape of a real community rather than a popular
node. Every k-truss is inside the k minus one core, and the truss number
of an edge, the largest k for which it survives, orders edges by how
embedded they are. The decomposition peels like the core: compute every
edge's support by intersecting its endpoints' neighborhoods, repeatedly
remove the edge of least support, recording its truss number as the
current level plus two, and lower the support of every edge that shared
a triangle with it. The engine computes truss numbers for every edge,
extracts any k-truss, reports the maximum truss number, and sets it
beside the degeneracy, because a graph whose truss number is far below
its core number plus one is a graph of hubs with unacquainted
neighbors, and one where they nearly match is a graph of tight groups.
"""

from __future__ import annotations

from mesh.errors import Invalid
from mesh.graph import Graph
from mesh.kcore import KCore


class KTruss:
    def __init__(self, graph: Graph) -> None:
        if graph.directed:
            raise Invalid("k-truss decomposition is defined for an undirected graph")
        self.graph = graph
        self._nbrs = {n: set(graph.neighbors(n)) for n in graph.nodes()}
        self.truss: dict[frozenset[str], int] = {}
        self._peel()

    def _peel(self) -> None:
        support = {
            frozenset((u, v)): len(self._nbrs[u] & self._nbrs[v])
            for u, v, _w in self.graph.edges()
        }
        alive = {n: set(nb) for n, nb in self._nbrs.items()}
        level = 0
        while support:
            # the edge of least support goes; its level plus two is its truss number
            edge = min(support, key=lambda e: (support[e], sorted(e)))
            level = max(level, support[edge])
            self.truss[edge] = level + 2
            u, v = tuple(edge)
            del support[edge]
            for w in alive[u] & alive[v]:
                for other in (frozenset((u, w)), frozenset((v, w))):
                    if other in support:
                        support[other] -= 1
            alive[u].discard(v)
            alive[v].discard(u)

    def k_truss(self, k: int) -> Graph:
        if k < 2:
            raise Invalid("k must be at least two; a 2-truss is every edge")
        g = Graph()
        for n in self.graph.nodes():
            g.add_node(n)
        for edge, t in self.truss.items():
            if t >= k:
                u, v = sorted(edge)
                g.add_edge(u, v, self.graph.weight(u, v))
        return g

    def truss_number(self) -> int:
        return max(self.truss.values(), default=0)

    def note(self) -> str:
        core = KCore(self.graph).degeneracy()
        return (
            f"truss number {self.truss_number()} against core number {core}; far below "
            "core plus one is hubs with unacquainted neighbors, close to it is tight groups"
        )
