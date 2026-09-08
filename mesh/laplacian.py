"""Graph Laplacian: the matrix whose null space counts the components.

The Laplacian is the degree matrix minus the adjacency matrix: each
node's degree on the diagonal, minus one wherever an edge runs, zero
elsewhere. Every row sums to zero, which is the same as saying the
constant vector is in its null space, and that single fact unfolds into
most of spectral graph theory. The Laplacian's quadratic form, x
transpose L x, equals the sum over edges of the squared difference of x
at the two ends, so it is zero exactly when x is constant on every
connected component, and the dimension of the null space, the nullity,
is exactly the number of connected components. That is a theorem an
engine can check by two unrelated methods: count components by union-
find, compute the Laplacian's rank by Gaussian elimination, and confirm
that nodes minus rank equals the component count. The normalized
Laplacian, identity minus the adjacency scaled by one over the square
root of the degree product at each end, has eigenvalues in zero to two,
with two attained exactly when the graph has a bipartite component,
and it is the form spectral clustering prefers because it does not let
high-degree nodes dominate. The engine builds both matrices in a fixed
node order, computes the rank of the plain Laplacian by elimination
with partial pivoting and a tolerance, derives the nullity, verifies the
row sums vanish and the quadratic form matches the edge-difference sum
for any vector, and reports the nullity beside the union-find component
count, because when the two agree the linear algebra and the traversal
have confirmed each other, and when they differ the elimination
tolerance has been set wrong for the weights in play.
"""

from __future__ import annotations

import math

from mesh.connectedcomponents import ConnectedComponents
from mesh.errors import Invalid
from mesh.graph import Graph


class Laplacian:
    def __init__(self, graph: Graph, tolerance: float = 1e-9) -> None:
        if graph.directed:
            raise Invalid("the Laplacian here is for undirected graphs")
        if graph.node_count() == 0:
            raise Invalid("an empty graph has no Laplacian")
        self.graph = graph
        self.nodes = graph.nodes()
        self.tolerance = tolerance
        self._pos = {n: i for i, n in enumerate(self.nodes)}
        self.matrix = self._plain()

    def _plain(self) -> list[list[float]]:
        n = len(self.nodes)
        lap = [[0.0] * n for _ in range(n)]
        for u, v, w in self.graph.edges():
            i, j = self._pos[u], self._pos[v]
            lap[i][i] += w
            lap[j][j] += w
            lap[i][j] -= w
            lap[j][i] -= w
        return lap

    def normalized(self) -> list[list[float]]:
        n = len(self.nodes)
        degree = [self.matrix[i][i] for i in range(n)]
        out = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    out[i][j] = 1.0 if degree[i] > 0 else 0.0
                elif self.matrix[i][j] != 0 and degree[i] > 0 and degree[j] > 0:
                    out[i][j] = self.matrix[i][j] / math.sqrt(degree[i] * degree[j])
        return out

    def rank(self) -> int:
        # Gaussian elimination with partial pivoting; tiny pivots count as zero
        a = [row[:] for row in self.matrix]
        n = len(a)
        rank = 0
        row = 0
        for col in range(n):
            pivot = max(range(row, n), key=lambda r: abs(a[r][col]), default=None)
            if pivot is None or abs(a[pivot][col]) < self.tolerance:
                continue
            a[row], a[pivot] = a[pivot], a[row]
            for r in range(row + 1, n):
                factor = a[r][col] / a[row][col]
                for c in range(col, n):
                    a[r][c] -= factor * a[row][c]
            rank += 1
            row += 1
            if row == n:
                break
        return rank

    def nullity(self) -> int:
        return len(self.nodes) - self.rank()

    def row_sums_vanish(self) -> bool:
        return all(abs(sum(row)) < self.tolerance for row in self.matrix)

    def quadratic_form(self, x: dict[str, float]) -> float:
        n = len(self.nodes)
        vec = [x[node] for node in self.nodes]
        return sum(vec[i] * self.matrix[i][j] * vec[j] for i in range(n) for j in range(n))

    def edge_difference_sum(self, x: dict[str, float]) -> float:
        return sum(w * (x[u] - x[v]) ** 2 for u, v, w in self.graph.edges())

    def note(self) -> str:
        components = ConnectedComponents(self.graph).count()
        verdict = "agree" if self.nullity() == components else "DISAGREE, check the tolerance"
        return (
            f"nullity {self.nullity()} against {components} component(s) by union-find: "
            f"the linear algebra and the traversal {verdict}"
        )
