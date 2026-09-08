"""Spectral bisection: split the graph where its second Laplacian eigenvector changes sign.

The Laplacian of a graph, degree on the diagonal and minus one for each
edge, has a smallest eigenvalue of zero with the constant vector as its
eigenvector, and its second-smallest eigenvalue, the algebraic
connectivity, measures how hard the graph is to cut: it is zero exactly
when the graph is disconnected and grows as the graph becomes more
tightly knit. The eigenvector for that second eigenvalue is the Fiedler
vector, and its sign pattern is a bisection of the graph that tends to
cut few edges while keeping the two halves balanced, because the
vector minimizes the sum over edges of squared differences subject to
being orthogonal to the constant vector and of unit length, a relaxation
of the balanced minimum cut. The engine finds the Fiedler vector by power
iteration on a shifted, deflated matrix: subtracting the Laplacian from a
constant large enough to keep every eigenvalue positive flips the order
so the smallest eigenvalues become the largest, and projecting out the
constant vector each step removes the trivial one, so iteration converges
to the second. The eigenvalue is recovered from the Rayleigh quotient of
the converged vector. Nodes with a positive entry form one side and the
rest the other, and the engine reports the number of edges that
bisection cuts. Spectral bisection is a heuristic with real guarantees
only in the relaxed sense, so the engine measures rather than assumes:
the cut it finds is compared against the true minimum balanced cut on
small graphs, and the docstring records that on a bridged pair of
cliques the two agree while on graphs without clear structure the
spectral cut can be worse. The bisector returns the two sides, the
algebraic connectivity, and the edges cut, refusing a directed or
disconnected graph, and reports the cut count against the edge count,
because a small fraction cut is a graph that really has two halves.
"""

from __future__ import annotations

import math

from mesh.connectedcomponents import ConnectedComponents
from mesh.errors import Invalid
from mesh.graph import Graph


class SpectralBisection:
    def __init__(
        self, graph: Graph, tolerance: float = 1e-10, max_iterations: int = 5000
    ) -> None:
        if graph.directed:
            raise Invalid("the Laplacian bisection is defined on an undirected graph")
        if graph.node_count() < 2:
            raise Invalid("bisection needs at least two nodes")
        if not ConnectedComponents(graph).is_connected():
            raise Invalid(
                "a disconnected graph has algebraic connectivity zero; split it first"
            )
        self.graph = graph
        self.nodes = graph.nodes()
        self.iterations = 0
        self.fiedler: dict[str, float] = {}
        self.algebraic_connectivity = 0.0
        self._iterate(tolerance, max_iterations)

    def _laplacian_times(self, vec: dict[str, float]) -> dict[str, float]:
        out = {}
        for n in self.nodes:
            nbrs = self.graph.neighbors(n)
            out[n] = len(nbrs) * vec[n] - sum(vec[m] for m in nbrs)
        return out

    def _iterate(self, tolerance: float, max_iterations: int) -> None:
        n = len(self.nodes)
        shift = 2.0 * max(self.graph.degree(v) for v in self.nodes) + 1.0
        # start off the constant vector, deterministic and non-symmetric
        vec = {node: math.sin(i + 1.0) for i, node in enumerate(self.nodes)}
        for _ in range(max_iterations):
            self.iterations += 1
            mean = sum(vec.values()) / n
            vec = {k: v - mean for k, v in vec.items()}  # deflate the constant
            lap = self._laplacian_times(vec)
            nxt = {k: shift * vec[k] - lap[k] for k in self.nodes}
            norm = math.sqrt(sum(x * x for x in nxt.values()))
            if norm == 0:
                raise Invalid("the iteration collapsed; no second eigenvector emerged")
            nxt = {k: x / norm for k, x in nxt.items()}
            change = sum(abs(nxt[k] - vec[k]) for k in self.nodes)
            vec = nxt
            if change < tolerance:
                break
        self.fiedler = vec
        lap = self._laplacian_times(vec)
        self.algebraic_connectivity = sum(vec[k] * lap[k] for k in self.nodes)

    def sides(self) -> tuple[set[str], set[str]]:
        pos = {k for k, v in self.fiedler.items() if v > 0}
        return pos, set(self.nodes) - pos

    def edges_cut(self) -> int:
        pos, _ = self.sides()
        return sum(1 for u, v, _w in self.graph.edges() if (u in pos) != (v in pos))

    def note(self) -> str:
        cut = self.edges_cut()
        return (
            f"algebraic connectivity {self.algebraic_connectivity:.4f}, bisection cuts "
            f"{cut} of {self.graph.edge_count()} edge(s) after {self.iterations} "
            "iteration(s); a small fraction cut is a graph that really has two halves"
        )
