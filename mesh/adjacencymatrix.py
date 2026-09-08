"""Adjacency matrix: the dense form, whose powers count walks between nodes.

The adjacency matrix puts the graph in a square grid, one row and one
column per node, with a one where an edge runs from the row's node to the
column's and a zero elsewhere. It costs nodes-squared space regardless of
how many edges there are, which is why the adjacency list is the default
for sparse graphs, but the matrix pays for itself when the question is
algebraic rather than a traversal. The cleanest example is walk counting.
Multiply the matrix by itself and the entry at row i, column j becomes the
number of two-step walks from i to j, because it sums over every middle
node k the product of is-there-an-edge i to k and is-there-an-edge k to j.
The k-th power counts walks of exactly k steps, repeated edges and
revisits allowed, so the diagonal of the cube counts closed three-walks
and, divided by six for the undirected case, gives the triangle count
without any neighborhood intersection. The trace of a power, its diagonal
sum, counts closed walks and links to the graph's spectrum. Squaring a
matrix repeatedly, exponentiation by squaring, reaches the k-th power in a
logarithm of matrix multiplications, so long walk counts are cheap even
though each multiplication is nodes cubed. The matrix is also where an
undirected graph shows its symmetry, since the entry i j equals j i, and
where a directed graph shows its in and out degrees as column and row
sums. The engine builds the matrix from a graph in a fixed node order,
multiplies and powers it with plain nested loops, counts walks of a given
length between two nodes, derives the triangle count from the cube's
trace, and reports the density, ones over cells, because a density under
a few percent is the case where the list would have been the honest
representation and the matrix is mostly zeroes.
"""

from __future__ import annotations

from mesh.errors import Invalid, Missing
from mesh.graph import Graph


class AdjacencyMatrix:
    def __init__(self, graph: Graph) -> None:
        self.graph = graph
        self.order = graph.nodes()
        self._pos = {n: i for i, n in enumerate(self.order)}
        n = len(self.order)
        self.matrix: list[list[int]] = [[0] * n for _ in range(n)]
        for u, v, _w in graph.edges():
            self.matrix[self._pos[u]][self._pos[v]] = 1
            if not graph.directed:
                self.matrix[self._pos[v]][self._pos[u]] = 1

    @staticmethod
    def multiply(a: list[list[int]], b: list[list[int]]) -> list[list[int]]:
        n = len(a)
        out = [[0] * n for _ in range(n)]
        for i in range(n):
            for k in range(n):
                if a[i][k] == 0:
                    continue  # skip the zero rows that dominate a sparse matrix
                for j in range(n):
                    out[i][j] += a[i][k] * b[k][j]
        return out

    def power(self, k: int) -> list[list[int]]:
        if k < 0:
            raise Invalid("a matrix power needs a non-negative exponent")
        n = len(self.order)
        result = [[int(i == j) for j in range(n)] for i in range(n)]
        base = [row[:] for row in self.matrix]
        while k:  # exponentiation by squaring
            if k & 1:
                result = self.multiply(result, base)
            base = self.multiply(base, base)
            k >>= 1
        return result

    def walks(self, u: str, v: str, length: int) -> int:
        for n in (u, v):
            if n not in self._pos:
                raise Missing(f"node '{n}' is not in the graph")
        return self.power(length)[self._pos[u]][self._pos[v]]

    def trace(self, k: int) -> int:
        p = self.power(k)
        return sum(p[i][i] for i in range(len(self.order)))

    def triangles(self) -> int:
        if self.graph.directed:
            raise Invalid("the trace-over-six triangle count is for undirected graphs")
        return self.trace(3) // 6  # each triangle is 3 starts times 2 directions

    def is_symmetric(self) -> bool:
        n = len(self.order)
        return all(self.matrix[i][j] == self.matrix[j][i] for i in range(n) for j in range(n))

    def density(self) -> float:
        n = len(self.order)
        ones = sum(sum(row) for row in self.matrix)
        return ones / (n * n) if n else 0.0

    def note(self) -> str:
        return (
            f"{len(self.order)}x{len(self.order)} matrix at density "
            f"{self.density():.3f}; under a few percent the list was the honest "
            "form and this is mostly zeroes"
        )
