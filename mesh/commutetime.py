"""Hitting and commute times: how long a random walk takes to get there and back.

The hitting time from a to b is the expected number of steps a random
walk started at a needs to first reach b, and the commute time is the
round trip, hitting time there plus hitting time back. Hitting times
are not symmetric: a walk from a leaf of a star reaches the hub in one
step, and a walk from the hub reaches a chosen leaf only after
wandering through the others. Commute times are symmetric and, by a
result of Chandra and colleagues, equal twice the edge count times the
effective resistance between the two nodes when every edge is a unit
resistor, which ties the walk to the circuit. The hitting times to a
fixed target solve a linear system: the target's own time is zero,
and every other node's time is one plus the average of its neighbors'
times. The engine builds that system per target, solves it by
Gaussian elimination with partial pivoting, refuses a graph in which
the target is unreachable from somewhere, since the time would be
infinite, and checks the commute times against the resistance rule by
solving the Laplacian system for the resistance directly. Closed
forms pin it: on a path of n nodes the hitting time from one end to
the other is (n minus 1) squared, on a complete graph the hitting
time between any two nodes is n minus 1, and on a cycle of n the
commute time between neighbors is twice the edge count n times the
resistance (n minus 1) over n, which is 2(n minus 1). A directed
graph is refused since the walk here is on an undirected graph.
"""

from __future__ import annotations

from mesh.errors import Invalid, Unreachable
from mesh.graph import Graph


def _solve(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    # Gaussian elimination with partial pivoting on a copy
    n = len(rhs)
    a = [[*row, rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-12:
            raise Invalid("the system is singular; some node cannot reach the target")
        a[col], a[pivot] = a[pivot], a[col]
        for r in range(n):
            if r != col and a[r][col] != 0.0:
                factor = a[r][col] / a[col][col]
                for c in range(col, n + 1):
                    a[r][c] -= factor * a[col][c]
    return [a[i][n] / a[i][i] for i in range(n)]


class CommuteTime:
    def __init__(self, graph: Graph) -> None:
        if graph.directed:
            raise Invalid("hitting times here are for a walk on an undirected graph")
        self.graph = graph
        self.nodes = graph.nodes()
        self.index = {n: i for i, n in enumerate(self.nodes)}

    def _reachable_from(self, start: str) -> set[str]:
        seen = {start}
        stack = [start]
        while stack:
            node = stack.pop()
            for other in self.graph.neighbors(node):
                if other not in seen:
                    seen.add(other)
                    stack.append(other)
        return seen

    def hitting_times_to(self, target: str) -> dict[str, float]:
        if target not in self.index:
            raise Invalid(f"'{target}' is not a node of the graph")
        if len(self._reachable_from(target)) != len(self.nodes):
            raise Unreachable(f"some node never reaches '{target}', so its time is infinite")
        n = len(self.nodes)
        matrix = [[0.0] * n for _ in range(n)]
        rhs = [0.0] * n
        for node in self.nodes:
            i = self.index[node]
            if node == target:
                matrix[i][i] = 1.0
                continue
            degree = self.graph.degree(node)
            matrix[i][i] = 1.0
            for other in self.graph.neighbors(node):
                matrix[i][self.index[other]] -= 1.0 / degree
            rhs[i] = 1.0
        solution = _solve(matrix, rhs)
        return {node: solution[self.index[node]] for node in self.nodes}

    def hitting(self, a: str, b: str) -> float:
        return self.hitting_times_to(b)[a]

    def commute(self, a: str, b: str) -> float:
        return self.hitting(a, b) + self.hitting(b, a)

    def resistance(self, a: str, b: str) -> float:
        # solve L x = e_a - e_b with one node grounded; the resistance is x_a - x_b
        if a == b:
            return 0.0
        n = len(self.nodes)
        if len(self._reachable_from(a)) != len(self.nodes):
            raise Unreachable("resistance needs a connected graph")
        lap = [[0.0] * n for _ in range(n)]
        for u, v, _w in self.graph.edges():
            i, j = self.index[u], self.index[v]
            lap[i][i] += 1
            lap[j][j] += 1
            lap[i][j] -= 1
            lap[j][i] -= 1
        rhs = [0.0] * n
        rhs[self.index[a]] = 1.0
        rhs[self.index[b]] = -1.0
        ground = self.index[b]
        lap[ground] = [1.0 if c == ground else 0.0 for c in range(n)]
        rhs[ground] = 0.0
        x = _solve(lap, rhs)
        return x[self.index[a]] - x[self.index[b]]

    def by_resistance(self, a: str, b: str) -> float:
        return 2 * self.graph.edge_count() * self.resistance(a, b)

    def commute_matches_resistance(self, a: str, b: str) -> bool:
        return abs(self.commute(a, b) - self.by_resistance(a, b)) < 1e-6

    def note(self, a: str, b: str) -> str:
        there, back = self.hitting(a, b), self.hitting(b, a)
        return (
            f"hit {a}->{b} in {there:.3f} steps, back in {back:.3f}, commute "
            f"{there + back:.3f} against 2m R = {self.by_resistance(a, b):.3f}"
        )
