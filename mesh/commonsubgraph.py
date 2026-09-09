"""Maximum common subgraph: the largest shape two graphs share, found through their product.

Two molecules, two call graphs, or two social circles are compared by
what they have in common, and the maximum common induced subgraph is
the largest set of node pairs, one from each graph, such that the
pairs are adjacent in the first graph exactly when they are adjacent
in the second. Levi's reduction turns it into a clique problem: build
the modular product, whose nodes are pairs (a, x) with a from the
first graph and x from the second, and join two pairs when their
first coordinates differ, their second coordinates differ, and the
two coordinates are both adjacent or both non-adjacent in their own
graphs. A clique in that product is a set of pairs that agree on
every edge and non-edge, which is a common induced subgraph, and a
maximum clique is a maximum common subgraph. The engine builds the
product, runs the Bron-Kerbosch module on it, translates the largest
clique back into a node correspondence, reports the common size and
the number of shared edges, and reads a similarity as the common size
over the larger node count. A graph against itself has everything in
common, a graph against its complement shares only edgeless or
complete pieces, a path of three against a triangle shares two nodes
and one edge, and two disjoint relabelings of one shape share all of
it. The product has n times m nodes, so the module refuses inputs
whose product would exceed sixty-four nodes.
"""

from __future__ import annotations

from itertools import combinations

from mesh.bronkerbosch import BronKerbosch
from mesh.errors import Invalid
from mesh.graph import Graph


class CommonSubgraph:
    def __init__(self, first: Graph, second: Graph) -> None:
        if first.directed or second.directed:
            raise Invalid("the modular product here is built for undirected graphs")
        if first.node_count() * second.node_count() > 64:
            raise Invalid("the modular product would exceed sixty-four nodes")
        self.first = first
        self.second = second
        self.product = self._modular_product()
        self.pairs = self._largest()

    @staticmethod
    def name(a: str, x: str) -> str:
        return f"{a}|{x}"

    def _modular_product(self) -> Graph:
        g = Graph()
        pairs = [(a, x) for a in self.first.nodes() for x in self.second.nodes()]
        for a, x in pairs:
            g.add_node(self.name(a, x))
        for (a, x), (b, y) in combinations(pairs, 2):
            if a == b or x == y:
                continue
            if self.first.has_edge(a, b) == self.second.has_edge(x, y):
                g.add_edge(self.name(a, x), self.name(b, y))
        return g

    def _largest(self) -> list[tuple[str, str]]:
        if self.product.node_count() == 0:
            return []
        clique = BronKerbosch(self.product).largest()
        pairs = [tuple(node.split("|", 1)) for node in clique]
        return sorted((a, x) for a, x in pairs)

    def size(self) -> int:
        return len(self.pairs)

    def shared_edges(self) -> int:
        return sum(
            1
            for (a, _x), (b, _y) in combinations(self.pairs, 2)
            if self.first.has_edge(a, b)
        )

    def is_consistent(self) -> bool:
        # every pair of matched nodes must agree on adjacency in both graphs
        first_side = [a for a, _x in self.pairs]
        second_side = [x for _a, x in self.pairs]
        if len(set(first_side)) != len(first_side) or len(set(second_side)) != len(second_side):
            return False
        return all(
            self.first.has_edge(a, b) == self.second.has_edge(x, y)
            for (a, x), (b, y) in combinations(self.pairs, 2)
        )

    def similarity(self) -> float:
        larger = max(self.first.node_count(), self.second.node_count())
        return self.size() / larger if larger else 1.0

    def note(self) -> str:
        return (
            f"common subgraph of {self.size()} node(s) and {self.shared_edges()} edge(s); "
            f"similarity {self.similarity():.2f} over a product of {self.product.node_count()}"
        )
