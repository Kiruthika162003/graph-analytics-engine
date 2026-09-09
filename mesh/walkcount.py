"""Walk counts: how many walks of each length join two nodes, by integer matrix powers.

The k-th power of the adjacency matrix counts walks of length k
between every pair, and because the entries are integers the count
is exact where an eigenvalue sum is only close. This module holds
the adjacency matrix as integer rows, raises it to the requested
power by repeated multiplication, and answers three questions: the
number of walks of a length between two nodes, the number of closed
walks of that length from a node back to itself, and the trace, the
total closed walks, which the spectrum module computes as a sum of
eigenvalue powers and which must agree to rounding. Two small cases
have direct meanings that the tests hold: walks of length two
between distinct nodes are their common neighbors, and closed walks
of length three at a node are twice the triangles it corners. The
powers are kept so repeated questions at the same length cost
nothing, and a length below zero is refused. Directed graphs are
counted along arcs, where a walk of length two from a to c means an
arc into some b and an arc out of it, and the trace of the cube is
then three times the directed triangles, one closed walk per starting
node of each, which the tests state rather than leave implied.
Unknown node names are refused.
"""

from __future__ import annotations

from itertools import combinations

from mesh.errors import Invalid
from mesh.graph import Graph
from mesh.graphenergy import GraphEnergy

Matrix = list[list[int]]


class WalkCount:
    def __init__(self, graph: Graph) -> None:
        self.graph = graph
        self.nodes = graph.nodes()
        self.index = {n: i for i, n in enumerate(self.nodes)}
        n = len(self.nodes)
        base = [[0] * n for _ in range(n)]
        for u, v, _w in graph.edges():
            base[self.index[u]][self.index[v]] += 1
            if not graph.directed:
                base[self.index[v]][self.index[u]] += 1
        identity = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
        self.powers: list[Matrix] = [identity, base]

    def _multiply(self, a: Matrix, b: Matrix) -> Matrix:
        n = len(a)
        out = [[0] * n for _ in range(n)]
        for i in range(n):
            row = a[i]
            for k, x in enumerate(row):
                if x:
                    bk = b[k]
                    target = out[i]
                    for j in range(n):
                        target[j] += x * bk[j]
        return out

    def power(self, k: int) -> Matrix:
        if k < 0:
            raise Invalid("a walk length cannot be negative")
        while len(self.powers) <= k:
            self.powers.append(self._multiply(self.powers[-1], self.powers[1]))
        return self.powers[k]

    def walks(self, a: str, b: str, k: int) -> int:
        for name in (a, b):
            if name not in self.index:
                raise Invalid(f"'{name}' is not a node of the graph")
        return self.power(k)[self.index[a]][self.index[b]]

    def closed(self, a: str, k: int) -> int:
        return self.walks(a, a, k)

    def trace(self, k: int) -> int:
        m = self.power(k)
        return sum(m[i][i] for i in range(len(m)))

    def common_neighbors(self, a: str, b: str) -> int:
        return sum(1 for x in self.graph.neighbors(a) if self.graph.has_edge(b, x))

    def triangles_at(self, node: str) -> int:
        nbrs = list(self.graph.neighbors(node))
        return sum(1 for x, y in combinations(nbrs, 2) if self.graph.has_edge(x, y))

    def matches_spectrum(self, k: int) -> bool:
        if self.graph.directed:
            raise Invalid("the spectral count needs an undirected graph")
        return abs(self.trace(k) - GraphEnergy(self.graph).closed_walks(k)) < 1e-6

    def note(self, k: int) -> str:
        return f"{self.trace(k)} closed walk(s) of length {k} over {len(self.nodes)} node(s)"
