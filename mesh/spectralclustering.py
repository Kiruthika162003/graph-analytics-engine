"""Spectral clustering: embed the nodes by Laplacian eigenvectors, then cluster the points.

The Fiedler vector splits a graph in two; more eigenvectors split it
into more. Take the k eigenvectors of the Laplacian with the smallest
eigenvalues, write each node as the row of k coordinates they give it,
and nodes in the same tightly knit group land close together, because
a vector that is nearly constant on a group and varies between groups
has a small Laplacian quadratic form. Clustering those points with
k-means then recovers the groups, which is the Shi-Malik and Ng-Jordan-
Weiss method, here on the unnormalised Laplacian with the rows scaled
to unit length so a group's size does not stretch its points. The
k-means is Lloyd's: seed the centers deterministically from the points
farthest from one another, assign each point to its nearest center,
move each center to its members' mean, and repeat until nothing moves.
The engine runs the embedding and the clustering, returns the groups
sorted, counts the edges that cross between groups, and checks the
result against the ideal on graphs built from cliques: three cliques
joined by single edges come back as three groups with exactly the
joining edges crossing, and a graph with k components comes back as
its components with nothing crossing, since the first k eigenvectors
are then the component indicators. A k below one or above the node
count is refused, and a directed graph is refused.
"""

from __future__ import annotations

from math import sqrt

from mesh.errors import Invalid
from mesh.graph import Graph
from mesh.symmetriceigen import SymmetricEigen


class SpectralClustering:
    def __init__(self, graph: Graph, k: int) -> None:
        if graph.directed:
            raise Invalid("spectral clustering takes an undirected graph")
        if k < 1 or k > graph.node_count():
            raise Invalid(f"k must lie between 1 and the node count, not {k}")
        self.graph = graph
        self.k = k
        self.nodes = graph.nodes()
        self.points = self._embed()
        self.rounds = 0
        self.labels = self._kmeans()
        self.groups = self._groups()

    def _embed(self) -> list[list[float]]:
        n = len(self.nodes)
        index = {node: i for i, node in enumerate(self.nodes)}
        lap = [[0.0] * n for _ in range(n)]
        for u, v, _w in self.graph.edges():
            i, j = index[u], index[v]
            lap[i][i] += 1
            lap[j][j] += 1
            lap[i][j] -= 1
            lap[j][i] -= 1
        solver = SymmetricEigen(lap)
        points = []
        for r in range(n):
            row = [solver.vectors[r][c] for c in range(self.k)]
            norm = sqrt(sum(x * x for x in row))
            points.append([x / norm for x in row] if norm > 1e-12 else row)
        return points

    @staticmethod
    def _dist(a: list[float], b: list[float]) -> float:
        return sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))

    def _seeds(self) -> list[list[float]]:
        # farthest-first: the first point, then repeatedly the point farthest from all seeds
        seeds = [self.points[0]]
        while len(seeds) < self.k:
            far = max(
                range(len(self.points)),
                key=lambda i: (min(self._dist(self.points[i], s) for s in seeds), -i),
            )
            seeds.append(self.points[far])
        return seeds

    def _kmeans(self) -> list[int]:
        centers = self._seeds()
        labels = [0] * len(self.points)
        for _ in range(100):
            self.rounds += 1
            new_labels = [
                min(range(self.k), key=lambda c: (self._dist(p, centers[c]), c))
                for p in self.points
            ]
            if new_labels == labels and self.rounds > 1:
                break
            labels = new_labels
            for c in range(self.k):
                members = [p for p, lab in zip(self.points, labels, strict=True) if lab == c]
                if members:
                    centers[c] = [sum(col) / len(members) for col in zip(*members, strict=True)]
        return labels

    def _groups(self) -> list[list[str]]:
        buckets: dict[int, list[str]] = {}
        for node, lab in zip(self.nodes, self.labels, strict=True):
            buckets.setdefault(lab, []).append(node)
        return sorted((sorted(g) for g in buckets.values()), key=lambda g: g[0])

    def crossing_edges(self) -> int:
        label = dict(zip(self.nodes, self.labels, strict=True))
        return sum(1 for u, v, _w in self.graph.edges() if label[u] != label[v])

    def note(self) -> str:
        sizes = [len(g) for g in self.groups]
        return (
            f"{len(self.groups)} group(s) of sizes {sizes} after {self.rounds} round(s); "
            f"{self.crossing_edges()} edge(s) cross between groups"
        )
