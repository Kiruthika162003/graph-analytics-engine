"""Laplacian spectrum: algebraic connectivity, spanning trees, and the cut it recommends.

The Laplacian is degree on the diagonal and minus one for each edge,
and its eigenvalues read connectivity in a way the adjacency spectrum
does not. The smallest is always zero, with the all-ones vector; the
number of zeros is the number of connected components; and the second
smallest, Fiedler's algebraic connectivity, is positive exactly when
the graph is connected and grows as the graph becomes harder to cut.
Its eigenvector, the Fiedler vector, splits the nodes by sign into a
two-way partition that is close to the sparsest cut, which is the
seed of spectral clustering. Two identities pin the numbers. The
eigenvalues sum to twice the edge count, the trace. And the product of
the non-zero eigenvalues divided by n is the number of spanning trees,
which the matrix-tree module reaches through a determinant, so the
two must agree. A complete graph on n nodes has spectrum zero once and
n repeated n minus one times, a path on n has 2 minus 2 cos(pi k
over n), and a star on n nodes has zero, one repeated n minus 2
times, and n. The engine builds the Laplacian, runs the Jacobi solver,
exposes the component count, the algebraic connectivity, the Fiedler
partition, the tree count, and the identities, and refuses a directed
graph.
"""

from __future__ import annotations

from mesh.errors import Invalid
from mesh.graph import Graph
from mesh.kirchhoff import SpanningTreeCount
from mesh.symmetriceigen import SymmetricEigen


class LaplacianSpectrum:
    def __init__(self, graph: Graph, zero: float = 1e-8) -> None:
        if graph.directed:
            raise Invalid("the Laplacian spectrum is read on an undirected graph")
        self.graph = graph
        self.nodes = graph.nodes()
        self.zero = zero
        matrix = [[0.0] * len(self.nodes) for _ in self.nodes]
        index = {n: i for i, n in enumerate(self.nodes)}
        for u, v, _w in graph.edges():
            i, j = index[u], index[v]
            matrix[i][i] += 1
            matrix[j][j] += 1
            matrix[i][j] -= 1
            matrix[j][i] -= 1
        self.solver = SymmetricEigen(matrix) if self.nodes else None
        self.values = self.solver.values if self.solver else []

    def components(self) -> int:
        return sum(1 for x in self.values if abs(x) < self.zero)

    def algebraic_connectivity(self) -> float:
        if len(self.values) < 2:
            return 0.0
        return self.values[1]

    def is_connected(self) -> bool:
        return len(self.nodes) <= 1 or self.algebraic_connectivity() > self.zero

    def fiedler_partition(self) -> tuple[list[str], list[str]]:
        if self.solver is None or len(self.nodes) < 2:
            return list(self.nodes), []
        vec = [self.solver.vectors[r][1] for r in range(len(self.nodes))]
        left = sorted(n for n, x in zip(self.nodes, vec, strict=True) if x < 0)
        right = sorted(n for n, x in zip(self.nodes, vec, strict=True) if x >= 0)
        return left, right

    def cut_size(self) -> int:
        left, _right = self.fiedler_partition()
        inside = set(left)
        return sum(1 for u, v, _w in self.graph.edges() if (u in inside) != (v in inside))

    def spanning_trees(self) -> float:
        n = len(self.nodes)
        if n == 0:
            return 0.0
        if self.components() != 1:
            return 0.0
        product = 1.0
        for x in self.values[1:]:
            product *= x
        return product / n

    def trace_identity_holds(self) -> bool:
        return abs(sum(self.values) - 2 * self.graph.edge_count()) < 1e-7

    def agrees_with_kirchhoff(self) -> bool:
        if not self.nodes:
            return True
        return abs(self.spanning_trees() - SpanningTreeCount(self.graph).count) < 1e-6

    def note(self) -> str:
        left, right = self.fiedler_partition()
        trees = f"{self.spanning_trees():.0f} spanning tree(s)"
        return (
            f"{self.components()} component(s), algebraic connectivity "
            f"{self.algebraic_connectivity():.4f}, {trees}; Fiedler cut of "
            f"{self.cut_size()} edge(s) splits {len(left)} from {len(right)}"
        )
