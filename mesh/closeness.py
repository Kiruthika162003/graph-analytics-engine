"""Closeness centrality: how near a node is to everyone, and the harmonic fix.

Closeness measures a node by how short its paths to all the others are. The
classic definition is the reciprocal of the node's total shortest-path
distance to every other node, so a node in the center of a graph, a few
hops from everything, scores high, while a node at the far end of a long
tail, far from most things, scores low. It is the centrality of the
well-placed: not the most connected, not the busiest broker, but the one
that can reach the whole graph fastest. The classic form has a well-known
flaw. In a disconnected graph some distances are infinite, so every node's
total is infinite and every closeness is zero, which throws away all the
information about how well placed a node is within its own component. The
usual patch, computing over the reachable nodes only and scaling by the
reachable fraction, helps but still compares nodes across components
unevenly. The harmonic form fixes it cleanly: instead of the reciprocal of
the sum of distances, take the sum of the reciprocals of distances, with an
unreachable node contributing zero, which is what one over infinity is. The
harmonic sum is finite for every node in any graph, rewards many short
paths, and simply gives no credit for nodes that cannot be reached, so it
compares nodes fairly whether or not the graph is connected. Both are
computed by a breadth-first search from each node, nodes times edges in
total. The measure returns the classic closeness, the harmonic closeness,
and the top nodes by either, and it reports whether the graph was
disconnected, because that is exactly the case where the two forms diverge
and the harmonic one is the honest reading.
"""

from __future__ import annotations

from mesh.bfs import BFS
from mesh.errors import Invalid
from mesh.graph import Graph


class Closeness:
    def __init__(self, graph: Graph) -> None:
        if graph.node_count() < 2:
            raise Invalid("closeness needs at least two nodes to have a distance")
        self.graph = graph
        self.classic: dict[str, float] = {}
        self.harmonic: dict[str, float] = {}
        self.disconnected = False
        self._run()

    def _run(self) -> None:
        n = self.graph.node_count()
        for node in self.graph.nodes():
            search = BFS(self.graph, node)
            others = {v: d for v, d in search.distance.items() if v != node}
            if len(others) < n - 1:
                self.disconnected = True
            total = sum(others.values())
            reached = len(others)
            # classic: reciprocal of the distance sum, scaled to the reached share
            if total > 0:
                self.classic[node] = (reached / (n - 1)) * (reached / total)
            else:
                self.classic[node] = 0.0
            # harmonic: sum of reciprocal distances, unreachable contributes zero
            self.harmonic[node] = sum(1.0 / d for d in others.values()) / (n - 1)

    def top(self, k: int = 3, harmonic: bool = True) -> list[tuple[str, float]]:
        table = self.harmonic if harmonic else self.classic
        ranked = sorted(table.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:k]

    def note(self) -> str:
        node, score = self.top(1)[0]
        shape = "disconnected, harmonic is the honest reading" if self.disconnected \
            else "connected, classic and harmonic agree on the ranking"
        return f"most central '{node}' at harmonic {score:.3f}; graph is {shape}"
