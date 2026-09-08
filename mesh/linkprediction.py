"""Link prediction: which missing edges are most likely to appear next.

Given a network as it stands, which pairs of nodes that are not yet
connected are most likely to connect? Friend suggestions, citation
recommendations, and protein-interaction candidates all reduce to
scoring the absent pairs by how much the existing structure implies
them. The classic scores all rest on the same intuition, that two nodes
with many shared neighbors are close, and differ in how they weigh the
sharing. Common neighbors is the raw count of shared neighbors. Jaccard
divides that count by the size of the union of the two neighborhoods,
so two nodes sharing three of their three neighbors each score higher
than two hubs sharing three of a hundred. Adamic-Adar sums one over the
log of each shared neighbor's degree, so a shared neighbor who knows
only these two is strong evidence while a shared neighbor who knows
everyone is weak, the rare acquaintance counting for more than the
celebrity. Preferential attachment multiplies the two degrees, ignoring
sharing entirely and betting that busy nodes keep getting busier, which
is the right score on a graph that grew by preferential attachment and
a poor one elsewhere. None of these is a prediction in the sense of a
guarantee; they are rankings, and the honest way to judge one is to hide
some edges, score the absent pairs, and measure how many hidden edges
land near the top, which the engine supports by scoring against any
graph the caller passes. The predictor scores a pair by each measure,
ranks all absent pairs by a chosen one, refuses a pair that is already
an edge or a node outside the graph, and reports the top candidate with
its shared-neighbor count, because a top pair with no shared neighbor
was ranked by degree alone and the caller should know the score is
guessing from popularity rather than from structure.
"""

from __future__ import annotations

import math
from itertools import combinations

from mesh.errors import Invalid, Missing
from mesh.graph import Graph

_MEASURES = ("common", "jaccard", "adamic_adar", "preferential")


class LinkPrediction:
    def __init__(self, graph: Graph) -> None:
        if graph.directed:
            raise Invalid("these link scores are defined on an undirected graph")
        self.graph = graph
        self._nbrs = {n: set(graph.neighbors(n)) for n in graph.nodes()}

    def _check(self, u: str, v: str) -> None:
        for n in (u, v):
            if n not in self._nbrs:
                raise Missing(f"node '{n}' is not in the graph")
        if u == v:
            raise Invalid("a node cannot link to itself")

    def common(self, u: str, v: str) -> float:
        self._check(u, v)
        return float(len(self._nbrs[u] & self._nbrs[v]))

    def jaccard(self, u: str, v: str) -> float:
        self._check(u, v)
        union = self._nbrs[u] | self._nbrs[v]
        return len(self._nbrs[u] & self._nbrs[v]) / len(union) if union else 0.0

    def adamic_adar(self, u: str, v: str) -> float:
        self._check(u, v)
        # a shared neighbor of degree one would divide by log(1) = 0; such a
        # neighbor cannot be shared by two nodes, so degree is at least two
        return sum(1.0 / math.log(len(self._nbrs[w])) for w in self._nbrs[u] & self._nbrs[v])

    def preferential(self, u: str, v: str) -> float:
        self._check(u, v)
        return float(len(self._nbrs[u]) * len(self._nbrs[v]))

    def score(self, u: str, v: str, measure: str) -> float:
        if measure not in _MEASURES:
            raise Invalid(f"unknown measure '{measure}'; choose from {_MEASURES}")
        return getattr(self, measure)(u, v)

    def ranked(self, measure: str = "adamic_adar") -> list[tuple[str, str, float]]:
        # every absent pair, best first, ties broken by name for determinism
        out = []
        for u, v in combinations(sorted(self._nbrs), 2):
            if not self.graph.has_edge(u, v):
                out.append((u, v, self.score(u, v, measure)))
        out.sort(key=lambda t: (-t[2], t[0], t[1]))
        return out

    def note(self, measure: str = "adamic_adar") -> str:
        ranking = self.ranked(measure)
        if not ranking:
            return "no absent pairs: the graph is complete, nothing left to predict"
        u, v, s = ranking[0]
        shared = int(self.common(u, v))
        basis = "structure" if shared else "popularity alone, the score is guessing"
        return (
            f"top candidate {u}-{v} by {measure} at {s:.3f} with {shared} shared "
            f"neighbor(s), ranked by {basis}"
        )
