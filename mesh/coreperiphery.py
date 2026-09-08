"""Core-periphery: a dense center everyone reaches, and a sparse rim that reaches only inward.

Some networks are not made of communities at all. They have one dense
core whose members connect to each other and to everyone, and a
periphery whose members connect to the core but rarely to each other:
international trade, airline routes through hub airports, a company
where a few people are on every project. Borgatti and Everett made
that shape testable. The ideal core-periphery pattern is a matrix with
ones inside the core block and between core and periphery and zeros
inside the periphery block, and a proposed split into core and
periphery is scored by how well the real adjacency correlates with that
ideal. The best split is a search over subsets, hopeless in general,
so the engine takes the standard shortcut: sort nodes by degree, try
every cut point along that order as the core boundary, and keep the
cut with the highest correlation, since a core almost always consists of
the highest-degree nodes. The correlation is the Pearson correlation
between the two flattened matrices over all node pairs, one for a
perfect fit, near zero for a graph with no such structure, and it is
reported alongside the fit so the reader knows how much the label is
earned. A community-structured graph of several equal cliques scores
poorly on every cut, which is the honest outcome: the model does not
fit and the coefficient says so rather than the engine forcing a core.
The fitter returns the core, the periphery, the correlation, and reports
the core size and coefficient, because a large core with a low
coefficient is a graph the model does not describe, and a small core
with a high one is the hub-and-spoke shape the model was made for.
"""

from __future__ import annotations

import math

from mesh.errors import Invalid
from mesh.graph import Graph


class CorePeriphery:
    def __init__(self, graph: Graph) -> None:
        if graph.directed:
            raise Invalid("this core-periphery fit is for undirected graphs")
        if graph.node_count() < 3:
            raise Invalid("a core and a periphery need at least three nodes")
        self.graph = graph
        self.nodes = sorted(graph.nodes(), key=lambda n: (-graph.degree(n), n))
        self.core: set[str] = set()
        self.correlation = -1.0
        self._fit()

    def _correlation(self, core: set[str]) -> float:
        # Pearson correlation of the adjacency against the ideal pattern
        actual: list[float] = []
        ideal: list[float] = []
        nodes = self.nodes
        for i, u in enumerate(nodes):
            for v in nodes[i + 1 :]:
                actual.append(1.0 if self.graph.has_edge(u, v) else 0.0)
                ideal.append(0.0 if (u not in core and v not in core) else 1.0)
        n = len(actual)
        ma, mi = sum(actual) / n, sum(ideal) / n
        cov = sum((a - ma) * (b - mi) for a, b in zip(actual, ideal, strict=True))
        va = sum((a - ma) ** 2 for a in actual)
        vi = sum((b - mi) ** 2 for b in ideal)
        if va == 0 or vi == 0:
            return 0.0
        return cov / math.sqrt(va * vi)

    def _fit(self) -> None:
        # every cut point along the degree order is a candidate core boundary
        for size in range(1, len(self.nodes)):
            core = set(self.nodes[:size])
            score = self._correlation(core)
            if score > self.correlation + 1e-12:
                self.correlation = score
                self.core = core

    def periphery(self) -> set[str]:
        return set(self.graph.nodes()) - self.core

    def fits(self, threshold: float = 0.5) -> bool:
        return self.correlation >= threshold

    def note(self) -> str:
        verdict = "hub-and-spoke shape" if self.fits() else "the model does not describe it"
        return (
            f"core of {len(self.core)} with correlation {self.correlation:.2f} to the "
            f"ideal pattern: {verdict}"
        )
