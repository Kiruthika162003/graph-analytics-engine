"""Harmonic centrality: closeness that survives a disconnected graph.

Closeness centrality is the reciprocal of a node's total distance to
everyone else, and it breaks the moment the graph is disconnected,
because one unreachable node makes the total infinite and the score
zero for everyone. Harmonic centrality, proposed by Marchiori and
Latora and argued for by Boldi and Vigna, sums the reciprocals of the
distances instead, so an unreachable node contributes zero and the
rest still count. On a connected graph the two rankings usually agree
but do not have to, and the module keeps both so a caller can see
where they part. The score is normalised by n minus 1, so a node
adjacent to everyone scores one and an isolated node scores zero.
Distances are hop counts from a breadth-first sweep, or Dijkstra
distances when weights are asked for, and on a directed graph the
sweep follows arcs outward, which reads how well a node reaches, not
how well it is reached; the reverse graph gives the other reading.
The engine scores every node, ranks them, reports the top node, and
checks two closed forms: a star's hub scores one and each leaf scores
(1 plus (n minus 2) over 2) over (n minus 1), and every node of a
complete graph scores one. A single node or an empty graph scores
nothing, without division by zero.
"""

from __future__ import annotations

from collections import deque

from mesh.dijkstra import Dijkstra
from mesh.graph import Graph


class Harmonic:
    def __init__(self, graph: Graph, weighted: bool = False) -> None:
        self.graph = graph
        self.weighted = weighted
        self.scores = {n: self._score(n) for n in graph.nodes()}

    def _distances(self, start: str) -> dict[str, float]:
        if self.weighted:
            return dict(Dijkstra(self.graph, start).distance)
        dist: dict[str, float] = {start: 0}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for other in self.graph.neighbors(node):
                if other not in dist:
                    dist[other] = dist[node] + 1
                    queue.append(other)
        return dist

    def _score(self, node: str) -> float:
        n = self.graph.node_count()
        if n < 2:
            return 0.0
        dist = self._distances(node)
        total = sum(1 / d for m, d in dist.items() if m != node and d > 0)
        return total / (n - 1)

    def closeness(self, node: str) -> float:
        n = self.graph.node_count()
        dist = self._distances(node)
        if len(dist) < n or n < 2:
            return 0.0
        return (n - 1) / sum(d for m, d in dist.items() if m != node)

    def ranking(self) -> list[str]:
        return sorted(self.scores, key=lambda n: (-self.scores[n], n))

    def top(self) -> str | None:
        ranking = self.ranking()
        return ranking[0] if ranking else None

    def rankings_agree(self) -> bool:
        by_closeness = sorted(self.scores, key=lambda n: (-self.closeness(n), n))
        return by_closeness == self.ranking()

    def note(self) -> str:
        top = self.top()
        if top is None:
            return "no nodes to score"
        agree = "agrees with" if self.rankings_agree() else "differs from"
        return (
            f"harmonic top {top} at {self.scores[top]:.3f}; the ranking {agree} closeness"
        )
