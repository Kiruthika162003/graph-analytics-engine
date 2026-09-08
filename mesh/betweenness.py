"""Betweenness centrality: how often a node sits on the shortest paths of others.

Betweenness measures a node's importance as a broker. For every pair of
other nodes, take the fraction of their shortest paths that pass through
this node, and sum those fractions over all pairs; a node with high
betweenness lies on many of the routes others must take, and removing it
would reroute or sever a large share of the traffic. It is the centrality
that finds bridges between communities, the one person whom two otherwise
separate groups both go through, where degree centrality would only count
their immediate friends. Computing it naively means enumerating all
shortest paths between all pairs, which is hopeless. Brandes's algorithm
does it in one pass per source. From each source it runs a breadth-first
search that counts the number of shortest paths to every node, sigma, by
adding a node's count to each successor it reaches on a shortest path. Then
it walks the nodes back in reverse order of discovery accumulating a
dependency: each node's dependency is the sum over its successors of the
successor's share of the source's paths through it, sigma ratio times one
plus the successor's own dependency. Summing each node's dependency over
all sources gives its betweenness, in nodes-times-edges time for an
unweighted graph. In an undirected graph every pair is counted twice, once
from each end, so the totals are halved. The measure returns the score per
node, the top brokers, and a normalized version dividing by the number of
pairs so scores compare across graphs of different size. It refuses a graph
with fewer than three nodes, where no node can lie between two others. It
reports the top node and its share of the maximum possible, because a top
score near the maximum is a single choke point the whole graph routes
through, a structural fragility worth knowing.
"""

from __future__ import annotations

from collections import deque

from mesh.errors import Invalid
from mesh.graph import Graph


class Betweenness:
    def __init__(self, graph: Graph) -> None:
        if graph.node_count() < 3:
            raise Invalid("betweenness needs at least three nodes for one to lie between")
        self.graph = graph
        self.score: dict[str, float] = dict.fromkeys(graph.nodes(), 0.0)
        self._run()

    def _run(self) -> None:
        for source in self.graph.nodes():
            self._accumulate_from(source)
        if not self.graph.directed:
            # each undirected pair was counted from both ends
            for node in self.score:
                self.score[node] /= 2.0

    def _accumulate_from(self, source: str) -> None:
        order: list[str] = []
        preds: dict[str, list[str]] = {n: [] for n in self.graph.nodes()}
        sigma: dict[str, float] = dict.fromkeys(self.graph.nodes(), 0.0)
        dist: dict[str, int] = dict.fromkeys(self.graph.nodes(), -1)
        sigma[source] = 1.0
        dist[source] = 0
        queue: deque[str] = deque([source])
        while queue:
            v = queue.popleft()
            order.append(v)
            for w in self.graph.neighbors(v):
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    queue.append(w)
                if dist[w] == dist[v] + 1:
                    # v is on a shortest path to w: pass its path count along
                    sigma[w] += sigma[v]
                    preds[w].append(v)
        delta: dict[str, float] = dict.fromkeys(self.graph.nodes(), 0.0)
        while order:
            w = order.pop()
            for v in preds[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != source:
                self.score[w] += delta[w]

    def normalized(self) -> dict[str, float]:
        n = self.graph.node_count()
        pairs = (n - 1) * (n - 2)
        if not self.graph.directed:
            pairs /= 2
        return {node: s / pairs for node, s in self.score.items()}

    def top(self, k: int = 3) -> list[tuple[str, float]]:
        ranked = sorted(self.score.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:k]

    def note(self) -> str:
        node, share = max(self.normalized().items(), key=lambda kv: kv[1])
        return (
            f"top broker '{node}' at {share * 100:.0f}% of the maximum; a share "
            "near 100 is a single choke point the whole graph routes through"
        )
