"""Kirchhoff's theorem: count the spanning trees of a graph with one determinant.

How many different spanning trees does a graph have? For a small graph
one can enumerate subsets of edges and test each, but the count grows so
fast that enumeration is hopeless on anything real: a complete graph on n
nodes has n to the power n minus two spanning trees, and on twenty nodes
that is a number with twenty-three digits. Kirchhoff's matrix-tree theorem
gives the count exactly with linear algebra. Build the Laplacian, the
matrix whose diagonal holds each node's degree and whose off-diagonal
entries are minus one where an edge runs, zero otherwise, so every row
sums to zero. Delete any one row and the same column, and the determinant
of what remains is the number of spanning trees. It does not matter which
row and column are removed; every choice gives the same value, a
consequence of the rows summing to zero, and the theorem's proof runs
through the Cauchy-Binet expansion of the reduced incidence matrix times
its transpose, where each nonzero term corresponds to exactly one spanning
tree. The determinant is computed here by Gaussian elimination with
partial pivoting on floating point, then rounded, which is exact for the
integer answers of modest graphs but would accumulate error on very large
ones, a limit the engine states rather than hides. The counter builds the
Laplacian, reduces it, computes the determinant, returns the tree count,
and cross-checks the count from two different deleted rows to catch an
arithmetic slip. It reports the count against the complete-graph maximum
for the same node count, because the ratio says how much of the possible
connectivity the graph actually has, a tree scoring exactly one and a
complete graph scoring the maximum.
"""

from __future__ import annotations

from mesh.errors import Invalid
from mesh.graph import Graph


class SpanningTreeCount:
    def __init__(self, graph: Graph) -> None:
        if graph.directed:
            raise Invalid("the matrix-tree theorem here is for undirected graphs")
        if graph.node_count() < 1:
            raise Invalid("an empty graph has no spanning tree to count")
        self.graph = graph
        self.nodes = graph.nodes()
        self.laplacian = self._laplacian()
        self.count = self._count(drop=0)
        # a second reduction must agree; a mismatch is an arithmetic slip
        self.cross_check = self._count(drop=len(self.nodes) - 1)

    def _laplacian(self) -> list[list[float]]:
        n = len(self.nodes)
        pos = {node: i for i, node in enumerate(self.nodes)}
        lap = [[0.0] * n for _ in range(n)]
        for u, v, _w in self.graph.edges():
            i, j = pos[u], pos[v]
            lap[i][i] += 1
            lap[j][j] += 1
            lap[i][j] -= 1
            lap[j][i] -= 1
        return lap

    def _count(self, drop: int) -> int:
        n = len(self.nodes)
        if n == 1:
            return 1  # a single node is its own spanning tree
        reduced = [
            [self.laplacian[i][j] for j in range(n) if j != drop]
            for i in range(n)
            if i != drop
        ]
        return round(self._determinant(reduced))

    @staticmethod
    def _determinant(m: list[list[float]]) -> float:
        # Gaussian elimination with partial pivoting; sign flips on each swap
        a = [row[:] for row in m]
        n = len(a)
        det = 1.0
        for col in range(n):
            pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
            if abs(a[pivot][col]) < 1e-12:
                return 0.0
            if pivot != col:
                a[col], a[pivot] = a[pivot], a[col]
                det = -det
            det *= a[col][col]
            for r in range(col + 1, n):
                factor = a[r][col] / a[col][col]
                for c in range(col, n):
                    a[r][c] -= factor * a[col][c]
        return det

    def complete_graph_maximum(self) -> int:
        n = len(self.nodes)
        return n ** (n - 2) if n >= 2 else 1

    def note(self) -> str:
        return (
            f"{self.count} spanning tree(s) against a complete-graph maximum of "
            f"{self.complete_graph_maximum()}; a tree scores one, a complete graph "
            "the maximum, and the cross-check agrees: "
            f"{self.count == self.cross_check}"
        )
