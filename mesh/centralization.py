"""Centralization: how much a whole graph is dominated by its most central node.

A centrality scores nodes; a centralization scores the graph. Freeman
defined it as the sum, over nodes, of how far each falls short of the
top score, divided by the largest that sum can be on any graph with
the same node count, which for degree and betweenness is reached by a
star: one hub with everything, every leaf with the least. A star
scores one, a cycle or a complete graph scores zero because every
node ties, and everything else lands between, so the number says
whether a network has a center at all. The engine computes it for
degree, where the maximum sum is (n minus 1)(n minus 2), for
betweenness, where the star's hub carries every leaf pair and each of
the n minus 1 leaves falls (n minus 1)(n minus 2) over 2 short of it,
and for harmonic centrality, where the maximum is taken from
the star directly since its closed form is less tidy, by building the
star of the same size and measuring it. The three readings are not
required to agree, and the note prints all three so a reader sees
which sense of center a graph has. A graph of fewer than three nodes
has no room for a center and scores zero; a directed graph is refused
since the star bounds are for undirected graphs.
"""

from __future__ import annotations

from mesh.errors import Invalid
from mesh.factories import star
from mesh.graph import Graph
from mesh.harmonic import Harmonic
from mesh.stresscentrality import StressCentrality


class Centralization:
    def __init__(self, graph: Graph) -> None:
        if graph.directed:
            raise Invalid("centralization here uses the undirected star as the ceiling")
        self.graph = graph
        self.n = graph.node_count()

    @staticmethod
    def _shortfall(scores: dict[str, float]) -> float:
        top = max(scores.values(), default=0.0)
        return sum(top - s for s in scores.values())

    def degree(self) -> float:
        if self.n < 3:
            return 0.0
        scores = {v: float(self.graph.degree(v)) for v in self.graph.nodes()}
        return self._shortfall(scores) / ((self.n - 1) * (self.n - 2))

    def betweenness(self) -> float:
        if self.n < 3:
            return 0.0
        # the star hub carries (n-1)(n-2)/2 pairs and each of n-1 leaves falls that short
        scores = StressCentrality(self.graph).betweenness
        return self._shortfall(scores) / ((self.n - 1) ** 2 * (self.n - 2) / 2)

    def harmonic(self) -> float:
        if self.n < 3:
            return 0.0
        ceiling = self._shortfall(Harmonic(star(self.n - 1)).scores)
        return self._shortfall(Harmonic(self.graph).scores) / ceiling

    def most_central(self) -> str | None:
        if self.n == 0:
            return None
        return max(sorted(self.graph.nodes()), key=self.graph.degree)

    def note(self) -> str:
        return (
            f"centralization by degree {self.degree():.3f}, betweenness "
            f"{self.betweenness():.3f}, harmonic {self.harmonic():.3f}; "
            f"busiest node {self.most_central()}"
        )
