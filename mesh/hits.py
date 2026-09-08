"""HITS: hubs point to good authorities, and authorities are pointed to by good hubs.

On a directed graph there are two different ways to be important, and
HITS scores both at once. An authority is a node that many good hubs point
to, the page that is the answer; a hub is a node that points to many good
authorities, the page that is the directory of answers. The two
definitions lean on each other, so the scores are found by mutual
reinforcement: set every hub and authority score to one, then repeat two
updates, each node's authority becomes the sum of the hub scores of the
nodes pointing at it, and each node's hub becomes the sum of the authority
scores of the nodes it points at, normalizing both vectors after every
round. The iteration converges to the principal eigenvectors of the
matrix products A-transpose-A for authorities and A-A-transpose for hubs,
which is why it settles where a single adjacency eigenvector on a
directed graph often would not. HITS makes the distinction PageRank
blurs. PageRank gives a single score that rewards being pointed to by
important nodes; HITS separates the pointer from the pointed-to, so a node
that links generously but is never linked scores as a strong hub and a
weak authority, and a node that is cited by everyone but cites no one is
the reverse. On a bipartite structure of directories and answers the two
rankings are entirely different lists. The scorer iterates until both
vectors stop changing, returns hub and authority per node normalized to
unit length, the top of each, and reports how many nodes are strong on
one axis and weak on the other, because that count is the structure HITS
exists to expose and a graph where the two rankings coincide is one where
PageRank alone would have done.
"""

from __future__ import annotations

import math

from mesh.errors import Invalid
from mesh.graph import Graph


class HITS:
    def __init__(
        self, graph: Graph, tolerance: float = 1e-10, max_iterations: int = 1000
    ) -> None:
        if not graph.directed:
            raise Invalid("hubs and authorities are a directed-graph notion")
        if graph.edge_count() == 0:
            raise Invalid("with no edges nothing points anywhere; no scores to find")
        self.graph = graph
        self._in: dict[str, list[str]] = {n: [] for n in graph.nodes()}
        for u, v, _w in graph.edges():
            self._in[v].append(u)
        self.iterations = 0
        self.hub, self.authority = self._iterate(tolerance, max_iterations)

    @staticmethod
    def _normalized(vec: dict[str, float]) -> dict[str, float]:
        norm = math.sqrt(sum(x * x for x in vec.values()))
        return {n: x / norm for n, x in vec.items()} if norm else vec

    def _iterate(
        self, tolerance: float, max_iterations: int
    ) -> tuple[dict[str, float], dict[str, float]]:
        nodes = self.graph.nodes()
        hub = dict.fromkeys(nodes, 1.0)
        auth = dict.fromkeys(nodes, 1.0)
        for _ in range(max_iterations):
            self.iterations += 1
            # authority: sum of hubs pointing in; hub: sum of authorities pointed at
            new_auth = self._normalized({n: sum(hub[u] for u in self._in[n]) for n in nodes})
            new_hub = self._normalized(
                {n: sum(new_auth[v] for v in self.graph.neighbors(n)) for n in nodes}
            )
            change = sum(abs(new_hub[n] - hub[n]) + abs(new_auth[n] - auth[n]) for n in nodes)
            hub, auth = new_hub, new_auth
            if change < tolerance:
                break
        return hub, auth

    def top_hubs(self, k: int = 3) -> list[tuple[str, float]]:
        return sorted(self.hub.items(), key=lambda kv: (-kv[1], kv[0]))[:k]

    def top_authorities(self, k: int = 3) -> list[tuple[str, float]]:
        return sorted(self.authority.items(), key=lambda kv: (-kv[1], kv[0]))[:k]

    def one_sided_count(self, threshold: float = 0.1) -> int:
        # nodes strong on exactly one axis: the structure HITS is for
        return sum(
            1
            for n in self.graph.nodes()
            if (self.hub[n] > threshold) != (self.authority[n] > threshold)
        )

    def note(self) -> str:
        return (
            f"top hub '{self.top_hubs(1)[0][0]}', top authority "
            f"'{self.top_authorities(1)[0][0]}', {self.one_sided_count()} node(s) "
            "strong on one axis only; when the two rankings coincide PageRank alone "
            "would have done"
        )
