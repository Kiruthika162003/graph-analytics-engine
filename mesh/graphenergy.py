"""Graph energy and the Estrada index: what the adjacency spectrum says about a graph.

Once the adjacency eigenvalues are in hand, several readings follow
that no walk or search produces directly. The energy is the sum of
the absolute eigenvalues, a quantity Gutman drew from Huckel theory
where it is the pi-electron energy of a molecule; a complete graph on
n nodes has energy 2(n minus 1), and an edgeless graph has none. The
Estrada index sums e to the power of each eigenvalue, weighting closed
walks of every length by the reciprocal factorial of the length, so
it reads how folded a graph is; it is at least n with equality only
when there are no edges. The spectral radius, the largest eigenvalue,
sits between the average degree and the maximum degree and is at
least the square root of the maximum degree, which the module checks.
Powers of the spectrum count closed walks: the eigenvalue squares sum
to twice the edge count, and the cubes sum to six times the triangle
count, because a closed walk of length three is a triangle walked in
one of six ways. The engine builds the adjacency matrix in node order,
runs the Jacobi solver, and exposes each reading beside the identity
that checks it, so a wrong eigenvalue shows up as a wrong edge count
rather than as a plausible number. A directed graph is refused since
its adjacency is not symmetric, and weights are ignored here because
the identities are statements about the unweighted adjacency.
"""

from __future__ import annotations

from itertools import combinations
from math import exp, sqrt

from mesh.errors import Invalid
from mesh.graph import Graph
from mesh.symmetriceigen import SymmetricEigen


class GraphEnergy:
    def __init__(self, graph: Graph) -> None:
        if graph.directed:
            raise Invalid("the adjacency spectrum of a directed graph is not symmetric")
        self.graph = graph
        self.nodes = graph.nodes()
        matrix = [
            [1.0 if graph.has_edge(a, b) else 0.0 for b in self.nodes] for a in self.nodes
        ]
        self.spectrum = SymmetricEigen(matrix).values if self.nodes else []

    def energy(self) -> float:
        return sum(abs(x) for x in self.spectrum)

    def estrada(self) -> float:
        return sum(exp(x) for x in self.spectrum)

    def spectral_radius(self) -> float:
        return max(self.spectrum, default=0.0)

    def radius_bounds_hold(self) -> bool:
        if not self.nodes:
            return True
        degrees = [self.graph.degree(n) for n in self.nodes]
        average = sum(degrees) / len(degrees)
        top = max(degrees)
        rho = self.spectral_radius()
        return average - 1e-9 <= rho <= top + 1e-9 and rho >= sqrt(top) - 1e-9

    def closed_walks(self, length: int) -> float:
        if length < 0:
            raise Invalid("walk length cannot be negative")
        return sum(x**length for x in self.spectrum)

    def triangles_from_spectrum(self) -> float:
        return self.closed_walks(3) / 6

    def triangles_by_search(self) -> int:
        has = self.graph.has_edge
        triples = combinations(self.nodes, 3)
        return sum(1 for a, b, c in triples if has(a, b) and has(b, c) and has(a, c))

    def walk_identities_hold(self) -> bool:
        edges_ok = abs(self.closed_walks(2) - 2 * self.graph.edge_count()) < 1e-7
        triangles_ok = abs(self.triangles_from_spectrum() - self.triangles_by_search()) < 1e-7
        return edges_ok and triangles_ok

    def note(self) -> str:
        return (
            f"energy {self.energy():.4f}, Estrada index {self.estrada():.4f}, spectral radius "
            f"{self.spectral_radius():.4f}; the spectrum counts {self.closed_walks(2):.0f} "
            f"closed two-walks and {self.triangles_from_spectrum():.0f} triangle(s)"
        )
