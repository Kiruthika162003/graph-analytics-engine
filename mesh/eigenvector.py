"""Eigenvector centrality: you matter in proportion to how much your neighbors matter.

Degree centrality counts a node's neighbors and treats them all alike.
Eigenvector centrality refines that by weighting each neighbor by its own
centrality, so a node connected to a few important nodes can outrank one
connected to many unimportant ones. The definition is recursive, a node's
score is proportional to the sum of its neighbors' scores, and the
resolution is the principal eigenvector of the adjacency matrix: the one
vector that, when the adjacency matrix acts on it, comes back as a multiple
of itself, with the largest such multiple. Power iteration finds it.
Start from a uniform vector, repeatedly replace each node's score with the
sum of its neighbors' scores, and renormalize so the vector keeps unit
length; each step amplifies the principal eigenvector's share relative to
every other, so the iteration converges to it, and the ratio the vector
grows by each step converges to the principal eigenvalue. Convergence
needs the graph to be connected, or the iteration splits across
components with no way to compare them, and it needs the graph not to be
bipartite, because on a bipartite graph the scores alternate between the
two sides every step and never settle; the engine checks both and refuses
rather than reporting whatever the iteration happened to be doing when it
hit its cap. This is the measure PageRank descends from, with damping and
teleport added precisely to remove those two restrictions on directed
graphs. The measure iterates until the vector stops changing, returns the
score per node normalized to unit length, the estimated eigenvalue, and
the top nodes, and reports the eigenvalue against the maximum degree,
because the principal eigenvalue of an adjacency matrix is bounded above
by the maximum degree and below by the average, so a value near the top
of that range is a graph whose high-degree nodes are all adjacent to one
another, a dense core.
"""

from __future__ import annotations

import math

from mesh.bipartite import Bipartite
from mesh.connectedcomponents import ConnectedComponents
from mesh.errors import Invalid
from mesh.graph import Graph


class EigenvectorCentrality:
    def __init__(
        self, graph: Graph, tolerance: float = 1e-10, max_iterations: int = 1000
    ) -> None:
        if graph.directed:
            raise Invalid("this eigenvector measure runs on an undirected graph")
        if graph.node_count() < 2:
            raise Invalid("eigenvector centrality needs at least two nodes")
        if not ConnectedComponents(graph).is_connected():
            raise Invalid("the graph must be connected for the iteration to converge")
        if Bipartite(graph).is_bipartite:
            raise Invalid("a bipartite graph makes the iteration alternate forever")
        self.graph = graph
        self.iterations = 0
        self.eigenvalue = 0.0
        self.score = self._power_iterate(tolerance, max_iterations)

    def _power_iterate(self, tolerance: float, max_iterations: int) -> dict[str, float]:
        nodes = self.graph.nodes()
        vec = dict.fromkeys(nodes, 1.0 / math.sqrt(len(nodes)))
        for _ in range(max_iterations):
            self.iterations += 1
            # each score becomes the sum of its neighbors' scores
            nxt = {n: sum(vec[m] for m in self.graph.neighbors(n)) for n in nodes}
            norm = math.sqrt(sum(x * x for x in nxt.values()))
            if norm == 0:
                raise Invalid("the iteration collapsed to zero; no edges to act on")
            nxt = {n: x / norm for n, x in nxt.items()}
            self.eigenvalue = norm  # growth per step converges to the eigenvalue
            change = sum(abs(nxt[n] - vec[n]) for n in nodes)
            vec = nxt
            if change < tolerance:
                break
        return vec

    def top(self, k: int = 3) -> list[tuple[str, float]]:
        ranked = sorted(self.score.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:k]

    def degree_bounds(self) -> tuple[float, float]:
        degrees = [self.graph.degree(n) for n in self.graph.nodes()]
        return sum(degrees) / len(degrees), float(max(degrees))

    def note(self) -> str:
        low, high = self.degree_bounds()
        node, _ = self.top(1)[0]
        return (
            f"eigenvalue {self.eigenvalue:.3f} within [{low:.2f}, {high:.0f}] after "
            f"{self.iterations} iteration(s), top '{node}'; near the top of the "
            "range is a dense core of mutually adjacent hubs"
        )
