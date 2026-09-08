"""Girvan-Newman: cut the edges that carry the most traffic, and watch the graph fall apart.

Where Louvain builds communities up by merging, Girvan-Newman finds them
by tearing the graph down. The observation is that an edge joining two
communities carries the shortest paths between them, all of them, so it
has high edge betweenness, while an edge inside a community is one of
many alternatives and carries little. Remove the edge of highest
betweenness, recompute betweenness on what remains, since removing an
edge reroutes the paths that used it, remove the new highest, and keep
going. The connected components at each stage are a partition, and the
sequence of partitions from one component down to singletons is a
hierarchy; the level to report is the one with the highest modularity,
which is where the removals have cut between communities and not yet
started cutting within them. Edge betweenness is computed by the same
Brandes accumulation as node betweenness, crediting each edge on the
backward pass with the dependency that flows across it, and the whole
method is expensive, a full betweenness pass per removal, cubic in the
edge count, which is why it is the method for hundreds of nodes rather
than millions and why Louvain exists. Its virtue is that it is
deterministic and its output is explainable: each split is a specific
edge that was carrying more between-group traffic than any other. The
detector runs removals until no edges remain or a cap is reached,
records the modularity after each, returns the best partition and the
edges removed to reach it, and reports the best modularity and the
removal count at which it peaked, because a peak reached after only a
few removals is a graph with a few clear seams, while a peak that
needed many is one whose communities were tangled together.
"""

from __future__ import annotations

from collections import deque

from mesh.connectedcomponents import ConnectedComponents
from mesh.errors import Invalid
from mesh.graph import Graph


class GirvanNewman:
    def __init__(self, graph: Graph, max_removals: int | None = None) -> None:
        if graph.directed:
            raise Invalid("Girvan-Newman here runs on an undirected graph")
        if graph.edge_count() == 0:
            raise Invalid("with no edges there is nothing to remove")
        self.graph = graph
        self.removed: list[tuple[str, str]] = []
        self.history: list[tuple[int, float]] = []  # (removals, modularity)
        self.best_partition: list[set[str]] = []
        self.best_modularity = -1.0
        self.best_at = 0
        self._run(max_removals if max_removals is not None else graph.edge_count())

    def _edge_betweenness(self, g: Graph) -> dict[frozenset[str], float]:
        score: dict[frozenset[str], float] = {
            frozenset((u, v)): 0.0 for u, v, _w in g.edges()
        }
        for source in g.nodes():
            order: list[str] = []
            preds: dict[str, list[str]] = {n: [] for n in g.nodes()}
            sigma = dict.fromkeys(g.nodes(), 0.0)
            dist = dict.fromkeys(g.nodes(), -1)
            sigma[source] = 1.0
            dist[source] = 0
            queue: deque[str] = deque([source])
            while queue:
                v = queue.popleft()
                order.append(v)
                for w in g.neighbors(v):
                    if dist[w] < 0:
                        dist[w] = dist[v] + 1
                        queue.append(w)
                    if dist[w] == dist[v] + 1:
                        sigma[w] += sigma[v]
                        preds[w].append(v)
            delta = dict.fromkeys(g.nodes(), 0.0)
            while order:
                w = order.pop()
                for v in preds[w]:
                    credit = (sigma[v] / sigma[w]) * (1.0 + delta[w])
                    score[frozenset((v, w))] += credit  # the edge carries it
                    delta[v] += credit
        return score

    def _modularity(self, g: Graph, parts: list[set[str]]) -> float:
        m = self.graph.edge_count()
        degree = {n: self.graph.degree(n) for n in self.graph.nodes()}
        label = {n: i for i, part in enumerate(parts) for n in part}
        total = 0.0
        for u in self.graph.nodes():
            for v in self.graph.nodes():
                if label[u] == label[v]:
                    actual = 1.0 if self.graph.has_edge(u, v) else 0.0
                    total += actual - degree[u] * degree[v] / (2 * m)
        del g
        return total / (2 * m)

    def _run(self, cap: int) -> None:
        work = Graph()
        for n in self.graph.nodes():
            work.add_node(n)
        for u, v, w in self.graph.edges():
            work.add_edge(u, v, w)
        self._record(work, 0)
        while work.edge_count() > 0 and len(self.removed) < cap:
            scores = self._edge_betweenness(work)
            top = max(scores, key=lambda e: (scores[e], sorted(e)))
            u, v = sorted(top)
            self.removed.append((u, v))
            work = self._without(work, u, v)
            self._record(work, len(self.removed))

    @staticmethod
    def _without(g: Graph, a: str, b: str) -> Graph:
        h = Graph()
        for n in g.nodes():
            h.add_node(n)
        for u, v, w in g.edges():
            if {u, v} != {a, b}:
                h.add_edge(u, v, w)
        return h

    def _record(self, work: Graph, removals: int) -> None:
        parts = ConnectedComponents(work).components()
        q = self._modularity(work, parts)
        self.history.append((removals, q))
        if q > self.best_modularity + 1e-12:
            self.best_modularity = q
            self.best_partition = parts
            self.best_at = removals

    def note(self) -> str:
        return (
            f"best modularity {self.best_modularity:.3f} with "
            f"{len(self.best_partition)} communit(ies) after {self.best_at} "
            "removal(s); an early peak is a few clear seams, a late one is tangled groups"
        )
