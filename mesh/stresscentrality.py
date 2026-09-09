"""Stress and load centrality: counting the shortest paths through a node, not their share.

Betweenness credits a node with the fraction of each pair's shortest
paths that pass through it. Stress centrality, older and blunter,
counts the shortest paths themselves: for each pair of other nodes,
add the number of shortest paths between them that pass through the
node. A node on the only route between two clusters scores their
product; a node on one of many parallel routes still scores every
path it lies on, so stress rewards nodes in dense, redundant regions
that betweenness discounts. Load centrality is a third reading, the
traffic a node carries when every pair sends one unit and each node
splits what it receives equally among its predecessors toward the
source, which is what a packet network without path counting does;
it equals betweenness on trees and diverges wherever two
predecessors of the same node carry different path counts. The engine
computes all three from one accumulation per source in the manner of
Brandes: a breadth-first search records path counts and
predecessors, and a reverse sweep pushes betweenness as fractional
dependencies, stress as the path count into a node times the number
of shortest-path continuations out of it, and load as split traffic.
Undirected pairs are counted once. A path's inner nodes score stress
equal to the product of the node counts on their two sides, a star's
hub scores the number of leaf pairs, and every node of a cycle scores
the same, which the tests check against a direct count over pairs.
"""

from __future__ import annotations

from collections import deque

from mesh.graph import Graph


class StressCentrality:
    def __init__(self, graph: Graph) -> None:
        self.graph = graph
        self.nodes = graph.nodes()
        self.stress = dict.fromkeys(self.nodes, 0.0)
        self.betweenness = dict.fromkeys(self.nodes, 0.0)
        self.load = dict.fromkeys(self.nodes, 0.0)
        for source in self.nodes:
            self._accumulate(source)
        if not graph.directed:
            for table in (self.stress, self.betweenness, self.load):
                for n in table:
                    table[n] /= 2

    def _accumulate(self, source: str) -> None:
        order: list[str] = []
        preds: dict[str, list[str]] = {n: [] for n in self.nodes}
        sigma = dict.fromkeys(self.nodes, 0)
        sigma[source] = 1
        dist = {source: 0}
        queue = deque([source])
        while queue:
            v = queue.popleft()
            order.append(v)
            for w in self.graph.neighbors(v):
                if w not in dist:
                    dist[w] = dist[v] + 1
                    queue.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    preds[w].append(v)
        delta = dict.fromkeys(self.nodes, 0.0)
        # continuations[v] counts shortest-path continuations from v to any descendant
        continuations = dict.fromkeys(self.nodes, 0)
        traffic = dict.fromkeys(self.nodes, 1.0)
        for w in reversed(order):
            for v in preds[w]:
                delta[v] += sigma[v] / sigma[w] * (1 + delta[w])
                continuations[v] += 1 + continuations[w]
                traffic[v] += traffic[w] / len(preds[w])
            if w != source:
                self.betweenness[w] += delta[w]
                self.stress[w] += sigma[w] * continuations[w]
                self.load[w] += traffic[w] - 1.0

    def ranking(self, measure: str = "stress") -> list[str]:
        table = {"stress": self.stress, "betweenness": self.betweenness, "load": self.load}
        if measure not in table:
            raise KeyError(f"no measure called '{measure}'")
        scores = table[measure]
        return sorted(scores, key=lambda n: (-scores[n], n))

    def load_matches_betweenness(self) -> bool:
        return all(abs(self.load[n] - self.betweenness[n]) < 1e-9 for n in self.nodes)

    def note(self) -> str:
        if not self.nodes:
            return "no nodes to score"
        top = self.ranking()[0]
        same = "matches" if self.load_matches_betweenness() else "differs from"
        return (
            f"most stressed {top} with {self.stress[top]:g} shortest path(s) through it; "
            f"load {same} betweenness"
        )
