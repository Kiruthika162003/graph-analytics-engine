"""Johnson: all-pairs paths on a sparse graph by reweighting negatives away.

Floyd-Warshall solves all-pairs shortest paths in nodes cubed, which is fine
for a dense graph but wasteful for a sparse one, where running Dijkstra
from every node would cost only nodes times edges times a logarithm.
Dijkstra cannot handle negative edges, though, and Johnson's algorithm is
the trick that lets it. Add a new virtual node with a zero-weight edge to
every existing node and run Bellman-Ford from it once; the resulting
distance to each node is a potential, h. Reweight every edge u to v as its
weight plus h of u minus h of v. Two things are true of the reweighted
graph. Every new weight is non-negative, because Bellman-Ford's distances
satisfy the triangle inequality, h of v is at most h of u plus the edge
weight, which is exactly the statement that the reweighted edge is at least
zero. And every path between a fixed pair of nodes changes by the same
amount, h of the start minus h of the end, because the intermediate
potentials telescope away, so the shortest path in the reweighted graph is
the shortest path in the original. Now Dijkstra runs from each node on
non-negative weights, and each distance is corrected back by subtracting
the start's potential and adding the end's. The single Bellman-Ford also
detects a negative cycle, which the potentials cannot exist under, so
Johnson refuses the same graphs Floyd-Warshall does. The solver builds the
potentials, reweights, runs Dijkstra per node, and returns the corrected
all-pairs distances, which must match Floyd-Warshall exactly on any graph
both accept. It reports the potential spread, the largest potential minus
the smallest, because a spread of zero means the graph had no negative
edges and the reweighting was a no-op that plain Dijkstra could have
skipped.
"""

from __future__ import annotations

import math

from mesh.bellmanford import BellmanFord
from mesh.dijkstra import Dijkstra
from mesh.errors import Unreachable
from mesh.graph import Graph


class Johnson:
    def __init__(self, graph: Graph) -> None:
        self.graph = graph
        self.potential = self._potentials()
        self.dist: dict[str, dict[str, float]] = self._all_pairs()

    def _potentials(self) -> dict[str, float]:
        # a virtual source with zero edges to everything gives h by Bellman-Ford
        augmented = Graph(directed=True)
        virtual = "\0virtual"
        augmented.add_node(virtual)
        for n in self.graph.nodes():
            augmented.add_node(n)
            augmented.add_edge(virtual, n, 0.0)
        for u, v, w in self.graph.edges():
            augmented.add_edge(u, v, w)
        bf = BellmanFord(augmented, virtual)  # raises Negative on a bad cycle
        return {n: bf.distance[n] for n in self.graph.nodes()}

    def _reweighted(self) -> Graph:
        h = self.potential
        g = Graph(directed=True)
        for n in self.graph.nodes():
            g.add_node(n)
        for u, v, w in self.graph.edges():
            g.add_edge(u, v, w + h[u] - h[v])  # non-negative by the triangle rule
        return g

    def _all_pairs(self) -> dict[str, dict[str, float]]:
        reweighted = self._reweighted()
        h = self.potential
        table: dict[str, dict[str, float]] = {}
        for u in self.graph.nodes():
            run = Dijkstra(reweighted, u)
            row = dict.fromkeys(self.graph.nodes(), math.inf)
            for v, d in run.distance.items():
                row[v] = d - h[u] + h[v]  # undo the telescoped shift
            table[u] = row
        return table

    def distance(self, u: str, v: str) -> float:
        if self.dist[u][v] == math.inf:
            raise Unreachable(f"'{v}' is unreachable from '{u}'")
        return self.dist[u][v]

    def potential_spread(self) -> float:
        values = list(self.potential.values())
        return max(values) - min(values) if values else 0.0

    def note(self) -> str:
        spread = self.potential_spread()
        return (
            f"potential spread {spread}; zero spread means no negative edges and "
            "the reweighting was a no-op plain Dijkstra could have skipped"
        )
