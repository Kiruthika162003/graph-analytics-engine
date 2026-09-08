"""Complement: flip every edge to a non-edge, and turn hard questions into their twins.

The complement of a graph has the same nodes and exactly the edges the
original lacks. It is a change of sign that pairs up problems: a clique
in the graph is an independent set in the complement, a vertex cover is
the nodes outside an independent set, and a graph whose complement is
isomorphic to itself, self-complementary, has exactly half of all
possible edges. The pairing is what makes the complement useful rather
than decorative. The engine already enumerates maximal cliques with
Bron-Kerbosch, so the maximum independent set of a graph, an NP-hard
problem with no direct method here, is the largest clique of the
complement, exact on any graph small enough for the clique enumeration.
The complement of a sparse graph is dense, so the trick is priced by the
density flip: a graph with a tenth of its possible edges has a complement
with nine tenths, where clique enumeration slows, and the engine reports
the density of both so the caller sees which side the work will fall
on. Identities pin the construction: the edge counts of a graph and its
complement sum to n choose two, a node's degrees in the two sum to n
minus one, and the complement of the complement is the original. The
module builds the complement, verifies those identities, extracts the
maximum independent set through the complement's largest clique, checks
that the set really is independent, and reports the independence number
beside the clique number, because their sum bounded by the node count is
a second identity and their product being small is a graph with neither
tight groups nor loose ones.
"""

from __future__ import annotations

from itertools import combinations

from mesh.bronkerbosch import BronKerbosch
from mesh.errors import Invalid
from mesh.graph import Graph


class Complement:
    def __init__(self, graph: Graph) -> None:
        if graph.directed:
            raise Invalid("this complement is for undirected graphs")
        self.graph = graph
        self.complement = self._build(graph)

    @staticmethod
    def _build(graph: Graph) -> Graph:
        g = Graph()
        nodes = sorted(graph.nodes())
        for n in nodes:
            g.add_node(n)
        for a, b in combinations(nodes, 2):
            if not graph.has_edge(a, b):
                g.add_edge(a, b)
        return g

    def identities_hold(self) -> bool:
        n = self.graph.node_count()
        if self.graph.edge_count() + self.complement.edge_count() != n * (n - 1) // 2:
            return False
        for v in self.graph.nodes():
            if self.graph.degree(v) + self.complement.degree(v) != n - 1:
                return False
        back = self._build(self.complement)
        return sorted(back.edges()) == sorted(
            (min(u, v), max(u, v), w) for u, v, w in self.graph.edges()
        )

    def maximum_independent_set(self) -> set[str]:
        # an independent set here is a clique in the complement
        if self.graph.node_count() == 0:
            return set()
        return set(BronKerbosch(self.complement).largest())

    def is_independent(self, nodes: set[str]) -> bool:
        return not any(self.graph.has_edge(a, b) for a, b in combinations(sorted(nodes), 2))

    def independence_number(self) -> int:
        return len(self.maximum_independent_set())

    def clique_number(self) -> int:
        return BronKerbosch(self.graph).clique_number()

    def density(self, graph: Graph) -> float:
        n = graph.node_count()
        return graph.edge_count() / (n * (n - 1) / 2) if n > 1 else 0.0

    def note(self) -> str:
        return (
            f"independence number {self.independence_number()} beside clique number "
            f"{self.clique_number()}; density {self.density(self.graph):.2f} flips to "
            f"{self.density(self.complement):.2f}, which is where the clique work lands"
        )
