"""Effective resistance: treat every edge as a resistor and measure between two nodes.

Replace each edge with a one-ohm resistor, or one over its weight for a
weighted edge, and the effective resistance between two nodes is what a
meter would read across them. It is a distance in its own right: it
obeys the triangle inequality, it equals the shortest path on a tree,
and on a graph with many parallel routes it drops below any single
path's length because parallel resistors combine as reciprocals. That
sensitivity to redundancy is why it measures robustness where shortest
path does not, two nodes joined by one route and two nodes joined by
ten routes of the same length have the same distance but very different
resistance. It also has an exact link to random walks: the expected
round trip between two nodes, the commute time, is twice the edge count
times their effective resistance. The computation is Kirchhoff's laws
solved as linear algebra. Ground one node by deleting its row and column
from the Laplacian, inject one unit of current at the source and draw it
at the target, and the potential difference the reduced system gives is
the resistance; the engine solves the reduced Laplacian by Gaussian
elimination with partial pivoting. The two textbook laws are its checks:
a path of unit edges reads its length in series, and two parallel paths
of the same length read half. The engine returns the resistance between
any pair, the commute time it implies, refuses a disconnected graph
where the resistance is infinite, and reports the resistance beside the
shortest-path distance, because the ratio is the redundancy between the
two nodes, one for a single route and smaller for every extra one.
"""

from __future__ import annotations

from mesh.connectedcomponents import ConnectedComponents
from mesh.dijkstra import Dijkstra
from mesh.errors import Invalid, Missing
from mesh.graph import Graph


class EffectiveResistance:
    def __init__(self, graph: Graph) -> None:
        if graph.directed:
            raise Invalid("resistance is defined on an undirected network")
        if graph.node_count() < 2:
            raise Invalid("resistance needs two nodes to measure between")
        if not ConnectedComponents(graph).is_connected():
            raise Invalid("a disconnected graph has infinite resistance between its pieces")
        self.graph = graph
        self.nodes = graph.nodes()
        self._pos = {n: i for i, n in enumerate(self.nodes)}

    def _laplacian(self) -> list[list[float]]:
        n = len(self.nodes)
        lap = [[0.0] * n for _ in range(n)]
        for u, v, w in self.graph.edges():
            if w <= 0:
                raise Invalid(f"edge {u}-{v} has weight {w}; conductance must be positive")
            i, j = self._pos[u], self._pos[v]
            lap[i][i] += w
            lap[j][j] += w
            lap[i][j] -= w
            lap[j][i] -= w
        return lap

    @staticmethod
    def _solve(a: list[list[float]], b: list[float]) -> list[float]:
        # Gaussian elimination with partial pivoting on an augmented system
        n = len(a)
        m = [[*row, b[i]] for i, row in enumerate(a)]
        for col in range(n):
            pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
            m[col], m[pivot] = m[pivot], m[col]
            for r in range(col + 1, n):
                factor = m[r][col] / m[col][col]
                for c in range(col, n + 1):
                    m[r][c] -= factor * m[col][c]
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            x[i] = (m[i][n] - sum(m[i][j] * x[j] for j in range(i + 1, n))) / m[i][i]
        return x

    def between(self, a: str, b: str) -> float:
        for n in (a, b):
            if n not in self._pos:
                raise Missing(f"node '{n}' is not in the graph")
        if a == b:
            return 0.0
        # ground b: drop its row and column, inject a unit of current at a
        lap = self._laplacian()
        ground = self._pos[b]
        keep = [i for i in range(len(self.nodes)) if i != ground]
        reduced = [[lap[i][j] for j in keep] for i in keep]
        current = [1.0 if i == self._pos[a] else 0.0 for i in keep]
        potential = self._solve(reduced, current)
        return potential[keep.index(self._pos[a])]

    def commute_time(self, a: str, b: str) -> float:
        total = sum(w for _u, _v, w in self.graph.edges())
        return 2.0 * total * self.between(a, b)

    def redundancy(self, a: str, b: str) -> float:
        distance = Dijkstra(self.graph, a).distance_to(b)
        return self.between(a, b) / distance if distance else 1.0

    def note(self, a: str, b: str) -> str:
        return (
            f"resistance {self.between(a, b):.3f} between '{a}' and '{b}', redundancy "
            f"{self.redundancy(a, b):.2f} of the shortest path; one is a single route, "
            "smaller is every extra one"
        )
