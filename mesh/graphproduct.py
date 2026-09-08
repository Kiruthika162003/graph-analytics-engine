"""Graph products: build a big graph from two small ones and know its shape in advance.

Three products share the same node set, every pair of a node from the
first graph and a node from the second, and differ in which pairs are
joined. The Cartesian product joins two pairs that agree in one
coordinate and are adjacent in the other, so a path times a path is a
grid and a cube is an edge times an edge times an edge. The tensor
product joins pairs adjacent in both coordinates at once, so a step
moves in both graphs together. The strong product is the union of the
two, and it is the shape of a king's moves on a board when both factors
are paths. Each product's counts follow from the factors: the
Cartesian product has n1 m2 plus n2 m1 edges and a degree that is the
sum of the coordinate degrees, the tensor product has 2 m1 m2 edges and
a degree that is the product, and the strong product adds the two
counts. Distances in the Cartesian product are the sum of the
coordinate distances, which is what makes it the natural product for
grids and hypercubes. The engine builds all three, names product nodes
by joining the coordinates with a comma, and reports the counts beside
the predicted counts so a reader can see the arithmetic hold. Directed
factors are refused; the products are defined for undirected graphs
here.
"""

from __future__ import annotations

from mesh.errors import Invalid
from mesh.graph import Graph


class GraphProduct:
    def __init__(self, first: Graph, second: Graph) -> None:
        if first.directed or second.directed:
            raise Invalid("graph products here take undirected factors")
        self.first = first
        self.second = second

    @staticmethod
    def name(a: str, b: str) -> str:
        return f"{a},{b}"

    def _nodes(self) -> Graph:
        g = Graph()
        for a in self.first.nodes():
            for b in self.second.nodes():
                g.add_node(self.name(a, b))
        return g

    def cartesian(self) -> Graph:
        g = self._nodes()
        for a in self.first.nodes():
            for u, v, _w in self.second.edges():
                g.add_edge(self.name(a, u), self.name(a, v))
        for b in self.second.nodes():
            for u, v, _w in self.first.edges():
                g.add_edge(self.name(u, b), self.name(v, b))
        return g

    def tensor(self) -> Graph:
        g = self._nodes()
        for u, v, _w in self.first.edges():
            for x, y, _z in self.second.edges():
                g.add_edge(self.name(u, x), self.name(v, y))
                g.add_edge(self.name(u, y), self.name(v, x))
        return g

    def strong(self) -> Graph:
        g = self.cartesian()
        for u, v, _w in self.tensor().edges():
            if not g.has_edge(u, v):
                g.add_edge(u, v)
        return g

    def predicted_edges(self, kind: str) -> int:
        n1, m1 = self.first.node_count(), self.first.edge_count()
        n2, m2 = self.second.node_count(), self.second.edge_count()
        if kind == "cartesian":
            return n1 * m2 + n2 * m1
        if kind == "tensor":
            return 2 * m1 * m2
        if kind == "strong":
            return n1 * m2 + n2 * m1 + 2 * m1 * m2
        raise Invalid(f"no product called '{kind}'")

    def note(self) -> str:
        parts = []
        for kind, build in (
            ("cartesian", self.cartesian),
            ("tensor", self.tensor),
            ("strong", self.strong),
        ):
            built = build().edge_count()
            parts.append(f"{kind} {built} edge(s), predicted {self.predicted_edges(kind)}")
        size = f"{self.first.node_count()}x{self.second.node_count()} nodes"
        return f"{size}; " + "; ".join(parts)
