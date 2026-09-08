"""Katz centrality: count every walk that reaches you, shorter walks counting more.

Eigenvector centrality only makes sense on a connected, non-bipartite
graph and gives zero to any node outside the largest strongly connected
core of a directed one. Katz centrality softens it. A node's Katz score
is the total number of walks of every length that end at it, each walk
of length k weighted by alpha to the k, so immediate neighbors count
fully, neighbors of neighbors count alpha squared, and long chains fade
away. With a small constant added so every node starts with something,
the score of each node is that constant plus alpha times the sum of its
in-neighbors' scores, a linear system whose solution is the vector
(I minus alpha A) inverse times the constant vector. The attenuation
alpha is the whole design decision. It must be strictly less than one
over the largest eigenvalue of the adjacency matrix, or the walk sum
diverges and the fixed-point iteration used to solve the system grows
without bound instead of converging; the engine estimates that
eigenvalue by power iteration and refuses an alpha at or above the
limit rather than returning the garbage a divergent iteration would
produce. Near the limit Katz approaches eigenvector centrality; near
zero it approaches in-degree, so alpha interpolates between counting
who points at you and counting who importantly points at you. The
measure solves the system by iterating the update from a uniform start
until it stops changing, returns the scores normalized to unit length,
the top nodes, and reports alpha as a fraction of its limit, because a
caller who chose alpha at ninety percent of the limit is measuring
something close to eigenvector centrality and one at ten percent is
measuring something close to degree.
"""

from __future__ import annotations

import math

from mesh.errors import Invalid
from mesh.graph import Graph


class Katz:
    def __init__(self, graph: Graph, alpha: float = 0.1, beta: float = 1.0) -> None:
        if graph.node_count() == 0:
            raise Invalid("an empty graph has nothing to score")
        if alpha <= 0:
            raise Invalid("alpha must be positive; zero would reduce Katz to a constant")
        self.graph = graph
        self.alpha = alpha
        self.beta = beta
        self._in: dict[str, list[str]] = {n: [] for n in graph.nodes()}
        for u, v, _w in graph.edges():
            self._in[v].append(u)
            if not graph.directed:
                self._in[u].append(v)
        self.spectral_radius = self._spectral_radius()
        limit = 1.0 / self.spectral_radius if self.spectral_radius > 0 else math.inf
        if alpha >= limit:
            raise Invalid(
                f"alpha {alpha} is at or above the convergence limit {limit:.4f} "
                "(one over the spectral radius); the walk sum would diverge"
            )
        self.iterations = 0
        self.score = self._solve()

    def _spectral_radius(self) -> float:
        nodes = self.graph.nodes()
        vec = dict.fromkeys(nodes, 1.0)
        radius = 0.0
        for _ in range(500):
            nxt = {n: sum(vec[m] for m in self._in[n]) for n in nodes}
            norm = math.sqrt(sum(x * x for x in nxt.values()))
            if norm == 0:
                return 0.0  # no walks at all: an edgeless or acyclic-thin graph
            nxt = {n: x / norm for n, x in nxt.items()}
            if abs(norm - radius) < 1e-12:
                return norm
            radius, vec = norm, nxt
        return radius

    def _solve(self) -> dict[str, float]:
        nodes = self.graph.nodes()
        vec = dict.fromkeys(nodes, self.beta)
        for _ in range(10000):
            self.iterations += 1
            nxt = {n: self.beta + self.alpha * sum(vec[m] for m in self._in[n]) for n in nodes}
            change = sum(abs(nxt[n] - vec[n]) for n in nodes)
            vec = nxt
            if change < 1e-12:
                break
        norm = math.sqrt(sum(x * x for x in vec.values()))
        return {n: x / norm for n, x in vec.items()}

    def top(self, k: int = 3) -> list[tuple[str, float]]:
        return sorted(self.score.items(), key=lambda kv: (-kv[1], kv[0]))[:k]

    def alpha_share(self) -> float:
        if self.spectral_radius == 0:
            return 0.0
        return self.alpha * self.spectral_radius

    def note(self) -> str:
        node, _ = self.top(1)[0]
        return (
            f"alpha at {self.alpha_share() * 100:.0f}% of its limit, top '{node}' after "
            f"{self.iterations} iteration(s); near the limit this is eigenvector "
            "centrality, near zero it is degree"
        )
