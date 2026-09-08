"""K-center: place k depots so that the farthest customer is as near as possible.

Choose k nodes as centers and assign every other node to its nearest
center; the cost is the largest of those distances, the worst-case trip
anyone has to make. The k-center problem asks for the centers that make
that worst case smallest, and it is the shape of placing fire stations,
cache servers, or ambulance bases when the promise is a maximum response
time rather than an average. It is NP-hard, and no polynomial algorithm
can do better than a factor of two unless P equals NP, so the honest
tool is the greedy that achieves exactly that factor. Farthest-first
traversal picks any node as the first center, then repeatedly picks as
the next center the node farthest from every center chosen so far,
until k are placed. The proof of the factor two is short: if the
optimal radius is r, the k plus one nodes consisting of the greedy
centers and the last farthest node are pairwise more than the greedy
radius apart, so two of them share an optimal center, and by the
triangle inequality the greedy radius is at most twice r. Distances
come from an all-pairs computation, so the method suits graphs of
modest size. The engine runs the traversal from a deterministic start,
returns the centers and the radius, computes the true optimum by trying
every k-subset when the graph is small enough, and reports the greedy
radius against that optimum, because the measured ratio is usually far
under two and the gap between the guarantee and the measurement is the
difference between a bound and a typical case.
"""

from __future__ import annotations

from itertools import combinations

from mesh.errors import Invalid
from mesh.floydwarshall import FloydWarshall
from mesh.graph import Graph

_EXACT_CAP = 12


class KCenter:
    def __init__(self, graph: Graph, k: int) -> None:
        if graph.directed:
            raise Invalid("this k-center placement is for undirected graphs")
        if k < 1 or k > graph.node_count():
            raise Invalid("k must be between one and the node count")
        self.graph = graph
        self.k = k
        self._dist = FloydWarshall(graph)
        for u in graph.nodes():
            for v in graph.nodes():
                if self._dist.dist[u][v] == float("inf"):
                    raise Invalid("the graph is disconnected; no center reaches everyone")
        self.centers = self._farthest_first()
        self.radius = self.radius_of(self.centers)

    def radius_of(self, centers: list[str]) -> float:
        return max(
            min(self._dist.dist[c][n] for c in centers) for n in self.graph.nodes()
        )

    def _farthest_first(self) -> list[str]:
        nodes = sorted(self.graph.nodes())
        # start at the node of least eccentricity: any start keeps the factor
        # two, but a first guess of starting at node 0 sat a lone center at
        # the end of a path, radius 6 against an optimum of 3, exactly the bound
        centers = [min(nodes, key=lambda n: (max(self._dist.dist[n][m] for m in nodes), n))]
        while len(centers) < self.k:
            # the node farthest from every center so far becomes the next center
            farthest = max(
                (n for n in nodes if n not in centers),
                key=lambda n: (min(self._dist.dist[c][n] for c in centers), n),
            )
            centers.append(farthest)
        return centers

    def optimum(self) -> float:
        if self.graph.node_count() > _EXACT_CAP:
            raise Invalid(f"exact search is capped at {_EXACT_CAP} nodes")
        return min(
            self.radius_of(list(subset))
            for subset in combinations(sorted(self.graph.nodes()), self.k)
        )

    def note(self) -> str:
        try:
            ratio = self.radius / self.optimum() if self.optimum() else 1.0
            measured = f"{ratio:.2f} times the optimum"
        except Invalid:
            measured = "optimum not computed past the cap"
        return (
            f"{self.k} center(s) {self.centers} with radius {self.radius}, {measured}; "
            "the guarantee is two, the typical case sits well under it"
        )
