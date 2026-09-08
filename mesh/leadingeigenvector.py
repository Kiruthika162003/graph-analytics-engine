"""Leading eigenvector: split a graph in two where modularity says the seam runs.

Louvain climbs modularity by moving nodes; Girvan-Newman descends by
cutting edges. Newman's spectral method reads the answer off a matrix.
The modularity matrix has, at each pair of nodes, the adjacency entry
minus the expected entry in a random graph with the same degrees, the
product of the two degrees over twice the edge count. Modularity of a
two-way split is then a quadratic form in the sign vector of the split,
and relaxing the signs to real numbers turns the best split into the
leading eigenvector of the modularity matrix: put each node on the side
of its entry's sign. If the leading eigenvalue is at or below zero, no
split improves on leaving the graph whole, which is the method's
built-in stopping rule and the honest signal that a graph has no
community structure to find. The engine finds the leading eigenvector
by power iteration shifted so every eigenvalue is positive, since the
modularity matrix has negative eigenvalues that would otherwise
dominate, recovers the eigenvalue from the Rayleigh quotient, and
converts the signs into a partition. It then does what the relaxation
alone does not: it measures the actual modularity of the sign split
against what Louvain reaches, because the spectral split is a
relaxation and its rounding can land well below a greedy climb on the
same graph. The splitter returns the two sides, the leading eigenvalue,
the modularity of the split, and reports whether the eigenvalue was
positive, because a non-positive one is the matrix saying the graph is
one piece, and forcing a split anyway would be reading structure into
noise.
"""

from __future__ import annotations

import math

from mesh.errors import Invalid
from mesh.graph import Graph


class LeadingEigenvector:
    def __init__(
        self, graph: Graph, tolerance: float = 1e-10, max_iterations: int = 5000
    ) -> None:
        if graph.directed:
            raise Invalid("this modularity split is for undirected graphs")
        if graph.edge_count() == 0:
            raise Invalid("with no edges there is no modularity to split on")
        self.graph = graph
        self.nodes = graph.nodes()
        self.m = graph.edge_count()
        self.degree = {n: graph.degree(n) for n in self.nodes}
        self.iterations = 0
        self.eigenvalue = 0.0
        self.vector = self._power(tolerance, max_iterations)

    def _modularity_times(self, vec: dict[str, float]) -> dict[str, float]:
        # B x = A x - k (k . x) / 2m, without forming the dense matrix
        weighted = sum(self.degree[n] * vec[n] for n in self.nodes) / (2 * self.m)
        out = {}
        for n in self.nodes:
            adjacency_part = sum(vec[m] for m in self.graph.neighbors(n))
            out[n] = adjacency_part - self.degree[n] * weighted
        return out

    def _power(self, tolerance: float, max_iterations: int) -> dict[str, float]:
        # shift by the largest possible magnitude so the top eigenvalue dominates
        shift = float(max(self.degree.values())) + 1.0
        vec = {n: math.cos(i + 0.5) for i, n in enumerate(self.nodes)}
        for _ in range(max_iterations):
            self.iterations += 1
            bx = self._modularity_times(vec)
            nxt = {n: bx[n] + shift * vec[n] for n in self.nodes}
            norm = math.sqrt(sum(x * x for x in nxt.values()))
            if norm == 0:
                break
            nxt = {n: x / norm for n, x in nxt.items()}
            change = sum(abs(nxt[n] - vec[n]) for n in self.nodes)
            vec = nxt
            if change < tolerance:
                break
        bx = self._modularity_times(vec)
        self.eigenvalue = sum(vec[n] * bx[n] for n in self.nodes)
        return vec

    def sides(self) -> tuple[set[str], set[str]]:
        pos = {n for n, x in self.vector.items() if x > 0}
        return pos, set(self.nodes) - pos

    def modularity(self) -> float:
        pos, _ = self.sides()
        total = 0.0
        for u in self.nodes:
            for v in self.nodes:
                if (u in pos) == (v in pos):
                    actual = 1.0 if self.graph.has_edge(u, v) else 0.0
                    total += actual - self.degree[u] * self.degree[v] / (2 * self.m)
        return total / (2 * self.m)

    def worth_splitting(self) -> bool:
        return self.eigenvalue > 1e-9

    def note(self) -> str:
        verdict = "a seam worth cutting" if self.worth_splitting() else "one piece, no seam"
        return (
            f"leading eigenvalue {self.eigenvalue:.3f} ({verdict}), split modularity "
            f"{self.modularity():.3f} after {self.iterations} iteration(s)"
        )
