"""Short cycle counts: triangles and four-cycles from traces, without enumeration.

Counting the short cycles of a graph is the quickest structural
fingerprint after the degrees: triangles measure clustering, four-cycles
measure the redundancy of two-hop paths, and the ratio of the two says
whether a network closes its triangles or leaves them as squares. Both
counts fall out of powers of the adjacency matrix. The trace of the cube
counts closed three-walks, and every triangle contributes six of them,
three starting corners times two directions, so the triangle count is
the trace over six. The trace of the fourth power counts closed four-
walks, but those include degenerate walks that go out and back twice
along the same edge or bounce along two edges, so the four-cycle count
is the trace of the fourth power, minus twice the sum of squared
degrees, plus the sum of degrees, all over eight, the eight being four
starting corners times two directions. The formula is checked by
enumeration on random graphs, because it is exactly the kind of
closed-form identity where an off-by-one in the degenerate-walk
correction survives a hand derivation and dies at the first test. The
engine computes the traces through the adjacency matrix module,
applies both formulas, counts both cycle lengths by direct enumeration
for graphs small enough, and reports the counts and the ratio of
four-cycles to triangles, because a graph with many squares and few
triangles is bipartite-like in its local structure while the reverse
is a graph of tight cliques.
"""

from __future__ import annotations

from itertools import combinations

from mesh.adjacencymatrix import AdjacencyMatrix
from mesh.errors import Invalid
from mesh.graph import Graph

_ENUM_CAP = 12


class CycleCount:
    def __init__(self, graph: Graph) -> None:
        if graph.directed:
            raise Invalid("these cycle counts are for undirected graphs")
        self.graph = graph
        self._matrix = AdjacencyMatrix(graph)
        degrees = [graph.degree(n) for n in graph.nodes()]
        self.triangles = self._matrix.trace(3) // 6
        # closed four-walks include out-and-back walks that are not cycles
        trace4 = self._matrix.trace(4)
        self.four_cycles = (trace4 - 2 * sum(d * d for d in degrees) + sum(degrees)) // 8

    def enumerate_triangles(self) -> int:
        self._check_cap()
        has = self.graph.has_edge
        return sum(
            1
            for a, b, c in combinations(self.graph.nodes(), 3)
            if has(a, b) and has(b, c) and has(a, c)
        )

    def enumerate_four_cycles(self) -> int:
        self._check_cap()
        count = 0
        for quad in combinations(sorted(self.graph.nodes()), 4):
            # three ways to arrange four nodes into a cycle
            a, b, c, d = quad
            for order in ((a, b, c, d), (a, b, d, c), (a, c, b, d)):
                if all(
                    self.graph.has_edge(order[i], order[(i + 1) % 4]) for i in range(4)
                ):
                    count += 1
        return count

    def _check_cap(self) -> None:
        if self.graph.node_count() > _ENUM_CAP:
            raise Invalid(f"enumeration is capped at {_ENUM_CAP} nodes")

    def square_to_triangle_ratio(self) -> float:
        return self.four_cycles / self.triangles if self.triangles else float("inf")

    def note(self) -> str:
        ratio = self.square_to_triangle_ratio()
        shape = "bipartite-like squares" if ratio > 2 else "tight triangles"
        return (
            f"{self.triangles} triangle(s), {self.four_cycles} four-cycle(s), ratio "
            f"{ratio:.2f}: {shape}"
        )
