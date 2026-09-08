"""Small world: high clustering and short paths at once, measured against a random twin.

A small-world network is one where your friends know each other, high
clustering, and yet anyone is a few hops from anyone, short paths. A
lattice has the first without the second; a random graph has the second
without the first; social networks, power grids, and neural wiring have
both, and Watts and Strogatz showed a few random shortcuts through a
lattice are all it takes. The measure is a comparison, not an absolute
number. Compute the graph's average clustering and its characteristic
path length, the mean shortest distance over all pairs. Build a random
graph with the same node and edge counts and compute the same two on
it. The small-world coefficient sigma is the clustering ratio divided
by the path-length ratio: a graph that clusters far more than its random
twin while keeping paths only slightly longer scores well above one, and
a random graph scores about one by construction. The comparison needs
the random twin to be connected for its path length to be finite, so
the engine draws twins until one is, up to a cap, and measures on the
largest component when the graph itself is disconnected, stating that
it did. Sigma has a known bias, it grows with graph size because random
clustering shrinks toward zero, so the engine also reports omega, the
path-length ratio to the random twin minus the clustering ratio to a
lattice twin, which sits near zero for a small world and at the extremes
for a lattice or a random graph. The measure returns both coefficients
and the four raw readings, and reports which regime the graph is in,
because a caller usually wants the word, small world or not, more than
the number.
"""

from __future__ import annotations

from mesh.connectedcomponents import ConnectedComponents
from mesh.errors import Invalid
from mesh.floydwarshall import FloydWarshall
from mesh.generators import erdos_renyi, watts_strogatz
from mesh.graph import Graph
from mesh.triangles import Triangles


class SmallWorld:
    def __init__(self, graph: Graph, seed: int = 0) -> None:
        if graph.directed:
            raise Invalid("the small-world measure is for undirected graphs")
        if graph.node_count() < 4 or graph.edge_count() < 3:
            raise Invalid("too small to compare against a random twin")
        self.graph = self._largest_component(graph)
        self.trimmed = self.graph.node_count() != graph.node_count()
        n, m = self.graph.node_count(), self.graph.edge_count()
        self.clustering = Triangles(self.graph).average_clustering()
        self.path_length = self._path_length(self.graph)
        self.random_twin = self._connected_random(n, m, seed)
        self.random_clustering = Triangles(self.random_twin).average_clustering()
        self.random_path_length = self._path_length(self.random_twin)
        self.lattice_clustering = self._lattice_clustering(n, m)

    @staticmethod
    def _largest_component(graph: Graph) -> Graph:
        biggest = ConnectedComponents(graph).components()[0]
        sub = Graph()
        for node in sorted(biggest):
            sub.add_node(node)
        for u, v, w in graph.edges():
            if u in biggest and v in biggest:
                sub.add_edge(u, v, w)
        return sub

    @staticmethod
    def _path_length(graph: Graph) -> float:
        fw = FloydWarshall(graph)
        nodes = graph.nodes()
        total = 0.0
        pairs = 0
        for i, a in enumerate(nodes):
            for b in nodes[i + 1 :]:
                total += fw.dist[a][b]
                pairs += 1
        return total / pairs if pairs else 0.0

    @staticmethod
    def _connected_random(n: int, m: int, seed: int) -> Graph:
        p = m / (n * (n - 1) / 2)
        for attempt in range(50):
            twin = erdos_renyi(n, p, seed=seed + attempt)
            if ConnectedComponents(twin).is_connected():
                return twin
        raise Invalid("no connected random twin found in fifty draws; the graph is too sparse")

    @staticmethod
    def _lattice_clustering(n: int, m: int) -> float:
        # a ring lattice with the same average degree, rounded down to even
        k = max(2, (2 * m // n) // 2 * 2)
        if k >= n:
            return 1.0
        return Triangles(watts_strogatz(n, k, 0.0)).average_clustering()

    def sigma(self) -> float:
        if not self.random_clustering:
            return float("inf")  # the twin has no triangles at all to compare against
        c_ratio = self.clustering / self.random_clustering
        l_ratio = self.path_length / self.random_path_length
        return c_ratio / l_ratio

    def omega(self) -> float:
        c_ratio = self.clustering / self.lattice_clustering if self.lattice_clustering else 0.0
        return self.random_path_length / self.path_length - c_ratio

    def verdict(self) -> str:
        if self.sigma() > 1.5 and abs(self.omega()) < 0.6:
            return "small world"
        if self.omega() > 0.6:
            return "random-like: short paths, no clustering"
        return "lattice-like: clustered but far apart"

    def note(self) -> str:
        scope = " on the largest component" if self.trimmed else ""
        return (
            f"sigma {self.sigma():.2f}, omega {self.omega():.2f}{scope}: {self.verdict()}; "
            f"clustering {self.clustering:.3f} vs random {self.random_clustering:.3f}, "
            f"path length {self.path_length:.2f} vs random {self.random_path_length:.2f}"
        )
