"""Cheeger constant: the sparsest cut, exactly, and the spectral bounds that bracket it.

The conductance of a node set S is the number of edges leaving S
divided by the smaller of the two sides' volumes, where volume is the
degree sum; the Cheeger constant of a graph is the smallest
conductance over every proper non-empty S. It is the bottleneck a
random walk must squeeze through, and finding it is NP-hard in
general, but on a small graph every subset can be tried. The Cheeger
inequality ties it to the second eigenvalue of the normalized
Laplacian, I minus D to the minus half A D to the minus half: half
that eigenvalue is at most the constant, and the constant is at most
the square root of twice that eigenvalue. The Fiedler vector of the
normalized Laplacian, scaled back by D to the minus half and swept
through its sorted order, gives a cut whose conductance also lands
within the upper bound, which is the constructive half of the
inequality. The engine builds the normalized Laplacian, runs the
Jacobi solver, takes the sweep cut with the best conductance along the
sorted eigenvector, tries every subset for the exact constant when the
graph has at most fourteen nodes, and checks the bounds. A complete
graph on n nodes has constant n over 2(n minus 1) rounded to the
smaller side, two cliques joined by one edge have a constant of one
over the smaller volume, and a cycle has 2 over n for even n when S is
half the cycle. A disconnected graph has constant zero and second
eigenvalue zero; a directed graph is refused.
"""

from __future__ import annotations

from itertools import combinations
from math import sqrt

from mesh.errors import Invalid
from mesh.graph import Graph
from mesh.symmetriceigen import SymmetricEigen


class Cheeger:
    def __init__(self, graph: Graph) -> None:
        if graph.directed:
            raise Invalid("conductance is read on an undirected graph")
        if any(graph.degree(n) == 0 for n in graph.nodes()):
            raise Invalid("an isolated node has no volume; remove it first")
        self.graph = graph
        self.nodes = graph.nodes()
        self.volume = sum(graph.degree(n) for n in self.nodes)
        self.index = {n: i for i, n in enumerate(self.nodes)}
        self.solver = SymmetricEigen(self._normalized()) if self.nodes else None
        self.lambda2 = self.solver.values[1] if self.solver and len(self.nodes) > 1 else 0.0

    def _normalized(self) -> list[list[float]]:
        n = len(self.nodes)
        m = [[0.0] * n for _ in range(n)]
        for i, node in enumerate(self.nodes):
            m[i][i] = 1.0
            for other in self.graph.neighbors(node):
                j = self.index[other]
                m[i][j] -= 1.0 / sqrt(self.graph.degree(node) * self.graph.degree(other))
        return m

    def conductance(self, inside: set[str]) -> float:
        if not inside or len(inside) == len(self.nodes):
            raise Invalid("conductance needs a proper non-empty side")
        cut = sum(1 for u, v, _w in self.graph.edges() if (u in inside) != (v in inside))
        vol = sum(self.graph.degree(n) for n in inside)
        return cut / min(vol, self.volume - vol)

    def sweep_cut(self) -> tuple[set[str], float]:
        # order nodes by the scaled Fiedler vector and take the best prefix
        if self.solver is None or len(self.nodes) < 2:
            raise Invalid("a sweep cut needs at least two nodes")
        vec = [
            self.solver.vectors[i][1] / sqrt(self.graph.degree(n))
            for i, n in enumerate(self.nodes)
        ]
        order = sorted(self.nodes, key=lambda n: (vec[self.index[n]], n))
        best: tuple[set[str], float] | None = None
        prefix: set[str] = set()
        for node in order[:-1]:
            prefix.add(node)
            phi = self.conductance(prefix)
            if best is None or phi < best[1]:
                best = (set(prefix), phi)
        assert best is not None
        return best

    def exact(self) -> tuple[set[str], float]:
        if len(self.nodes) > 14:
            raise Invalid("the exact constant tries every subset; keep it to fourteen nodes")
        if len(self.nodes) < 2:
            raise Invalid("the constant needs at least two nodes")
        best: tuple[set[str], float] | None = None
        first = self.nodes[0]
        others = self.nodes[1:]
        # fix the first node inside, so each cut is seen once
        for size in range(len(others) + 1):
            for chosen in combinations(others, size):
                inside = {first, *chosen}
                if len(inside) == len(self.nodes):
                    continue
                phi = self.conductance(inside)
                if best is None or phi < best[1]:
                    best = (inside, phi)
        assert best is not None
        return best

    def bounds_hold(self) -> bool:
        _side, h = self.exact()
        _sweep_side, sweep = self.sweep_cut()
        lower = self.lambda2 / 2
        upper = sqrt(2 * self.lambda2)
        return lower - 1e-9 <= h <= upper + 1e-9 and sweep <= upper + 1e-9

    def note(self) -> str:
        side, h = self.exact()
        _s, sweep = self.sweep_cut()
        return (
            f"Cheeger constant {h:.4f} on a side of {len(side)}, sweep cut {sweep:.4f}, "
            f"bracketed by {self.lambda2 / 2:.4f} and {sqrt(2 * self.lambda2):.4f}"
        )
