"""Diameter and center: how far apart the graph's farthest points are, and its middle.

The eccentricity of a node is the distance to the node farthest from it,
the worst case for a message sent from there. The diameter is the largest
eccentricity, the longest shortest path anywhere in the graph, the number
of hops that guarantees any two nodes can reach each other. The radius is
the smallest eccentricity, and the center is the set of nodes that attain
it, the best places to put something everyone needs to reach; the
periphery is the set attaining the diameter, the nodes on the outskirts.
Computing all of this exactly on an unweighted graph means a breadth-first
search from every node, nodes times edges, which is the honest cost and is
what this engine does for the exact answer. There is a much cheaper
estimate that is often exact and never overestimates: the double sweep.
Search from any node, take the farthest node found, search again from
there, and report that second eccentricity. On a tree the double sweep is
provably exact, because the farthest node from anywhere is always an
endpoint of a diameter path. On a general graph it is a lower bound that
can undershoot, and the engine computes both so the gap is visible rather
than assumed away: a first guess that the sweep is always right is refuted
by graphs where it lands short, and those graphs are the reason the exact
computation exists. A disconnected graph has infinite diameter by the
usual convention, so the engine refuses it and asks for a component
instead, rather than reporting a finite number that quietly ignores the
unreachable pairs. The measure returns eccentricities, diameter, radius,
center, periphery, and the double-sweep estimate, and reports the estimate
against the exact diameter, because when they differ the cheap number is
wrong and a caller relying on it should know by how much.
"""

from __future__ import annotations

from mesh.bfs import BFS
from mesh.errors import Invalid
from mesh.graph import Graph


class Diameter:
    def __init__(self, graph: Graph) -> None:
        if graph.node_count() == 0:
            raise Invalid("an empty graph has no diameter")
        self.graph = graph
        self.eccentricity: dict[str, int] = {}
        for node in graph.nodes():
            search = BFS(graph, node)
            if len(search.distance) != graph.node_count():
                raise Invalid(
                    "the graph is disconnected, so the diameter is infinite; "
                    "measure a single component instead"
                )
            self.eccentricity[node] = search.eccentricity()

    def diameter(self) -> int:
        return max(self.eccentricity.values())

    def radius(self) -> int:
        return min(self.eccentricity.values())

    def center(self) -> set[str]:
        r = self.radius()
        return {n for n, e in self.eccentricity.items() if e == r}

    def periphery(self) -> set[str]:
        d = self.diameter()
        return {n for n, e in self.eccentricity.items() if e == d}

    def double_sweep(self, start: str | None = None) -> int:
        # two BFS runs: a cheap lower bound, exact on trees, short elsewhere
        if start is None:
            start = self.graph.nodes()[0]
        first = BFS(self.graph, start)
        far = max(first.distance, key=lambda n: (first.distance[n], n))
        return BFS(self.graph, far).eccentricity()

    def sweep_gap(self) -> int:
        return self.diameter() - self.double_sweep()

    def note(self) -> str:
        gap = self.sweep_gap()
        verdict = "exact here" if gap == 0 else f"short by {gap}"
        return (
            f"diameter {self.diameter()}, radius {self.radius()}, center "
            f"{sorted(self.center())}; double sweep {verdict}, it never "
            "overestimates but on a general graph it can undershoot"
        )
