"""Chinese postman: the cheapest closed route that walks every street at least once.

A postal carrier must traverse every edge of a graph and return to the
start, and wants the shortest such route. If every node has even degree
the answer is an Eulerian circuit, walking every edge exactly once, and
the route costs exactly the total edge weight. When some nodes have odd
degree, no circuit uses each edge once, and some edges must be walked
twice. The question becomes which ones, and the answer has a clean
structure: the edges walked twice form paths that pair up the odd-degree
nodes, because doubling a path flips the parity of exactly its two ends.
Since there are always an even number of odd-degree nodes, they can be
perfectly paired, and the cheapest pairing under shortest-path distances
gives the minimum extra walking. The route cost is the total edge weight
plus the weight of that minimum perfect matching of the odd nodes. The
matching is a general, not bipartite, minimum-weight perfect matching,
which in full generality needs Edmonds's blossom machinery. This engine
takes the honest shortcut of solving it exactly by exhaustive pairing,
which is feasible for the handful of odd nodes a street network usually
has and is stated as a limit: with two odd nodes there is one pairing,
with four there are three, with ten there are nine hundred forty-five,
and the engine refuses beyond a cap rather than silently running for
hours. Pairwise distances come from Floyd-Warshall. The planner returns
the route cost, the odd-node pairing chosen, and the extra distance the
doubled paths add, and it reports that extra against the base weight,
because a postman whose extra walking is a large fraction of the route
is serving a network full of dead ends, where the parity fix is most of
the job.
"""

from __future__ import annotations

import math

from mesh.connectedcomponents import ConnectedComponents
from mesh.errors import Invalid
from mesh.floydwarshall import FloydWarshall
from mesh.graph import Graph

_ODD_CAP = 10


class ChinesePostman:
    def __init__(self, graph: Graph) -> None:
        if graph.directed:
            raise Invalid("this postman route is for undirected streets")
        if graph.edge_count() == 0:
            raise Invalid("no streets to walk")
        if not ConnectedComponents(graph).is_connected():
            raise Invalid("streets in separate components cannot share one route")
        self.graph = graph
        self.base = sum(w for _u, _v, w in graph.edges())
        self.odd = sorted(n for n in graph.nodes() if graph.degree(n) % 2 == 1)
        if len(self.odd) > _ODD_CAP:
            raise Invalid(
                f"{len(self.odd)} odd-degree nodes exceeds the exhaustive matching cap "
                f"of {_ODD_CAP}; a blossom matcher is needed beyond that"
            )
        self._dist = FloydWarshall(graph)
        self.pairing: list[tuple[str, str]] = []
        self.extra = self._match()

    def _match(self) -> float:
        # exhaustive minimum-weight perfect matching of the odd nodes
        best = math.inf
        best_pairing: list[tuple[str, str]] = []

        def recurse(remaining: list[str], chosen: list[tuple[str, str]], cost: float) -> None:
            nonlocal best, best_pairing
            if cost >= best:
                return  # this partial pairing already costs too much
            if not remaining:
                best, best_pairing = cost, list(chosen)
                return
            first = remaining[0]
            for partner in remaining[1:]:
                rest = [n for n in remaining if n not in (first, partner)]
                step = self._dist.distance(first, partner)
                recurse(rest, [*chosen, (first, partner)], cost + step)

        recurse(self.odd, [], 0.0)
        self.pairing = best_pairing
        return best if self.odd else 0.0

    def route_cost(self) -> float:
        return self.base + self.extra

    def pairings_considered(self) -> int:
        k = len(self.odd)
        return math.prod(range(1, k, 2)) if k else 1

    def note(self) -> str:
        share = self.extra / self.base if self.base else 0.0
        return (
            f"route costs {self.route_cost()}: base {self.base} plus {self.extra} of "
            f"doubled streets pairing {len(self.odd)} odd node(s) over "
            f"{self.pairings_considered()} pairing(s); extra at {share * 100:.0f}% "
            "of base is a network full of dead ends"
        )
