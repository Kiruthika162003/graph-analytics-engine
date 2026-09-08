"""Maximum cut: split the nodes so as many edges as possible cross the split.

The minimum cut asks for the fewest edges between two sides; the maximum
cut asks for the most, and the change of sign changes everything. Max
cut is NP-hard, one of Karp's original problems, so an exact answer on a
large graph is out of reach and the engine offers what is honestly
available: a local search with a proven guarantee. Start from any split
and repeatedly move a node to the other side whenever that increases the
number of crossing edges, until no single move helps. At that point every
node has at least half its edges crossing, because otherwise moving it
would gain, so the cut holds at least half of all edges, which is at
least half the optimum. That factor of one half is the guarantee, the
same one a uniformly random split achieves in expectation. A first guess
that the search would always find the full cut of a bipartite graph was
refuted by a six-cycle: a split whose same-side edges form a matching
leaves every node with gain zero, a local optimum at four crossing
edges out of six, and the search stops there. The remedy is the standard
one, several random starts with the best result kept, which reaches the
six on that cycle and in general climbs out of the shallow basins a
single start falls into, while the guarantee holds for every start. On
small graphs the engine's answer is compared against the true maximum
found by enumerating every bipartition, so the gap between heuristic and
optimum is measured rather than assumed. The optimizer returns the two
sides, the cut weight, the starts and moves taken, and reports the cut
against the total edge weight, because a cut near the total is a nearly
bipartite graph, where the max cut is almost everything, and a cut near
half is a dense graph where no split does much better than a coin flip.
"""

from __future__ import annotations

import random

from mesh.errors import Invalid
from mesh.graph import Graph


class MaxCut:
    def __init__(self, graph: Graph, seed: int = 0, restarts: int = 16) -> None:
        if graph.directed:
            raise Invalid("max cut here is on an undirected graph")
        if graph.node_count() < 2:
            raise Invalid("a cut needs at least two nodes")
        if restarts < 1:
            raise Invalid("at least one start is needed")
        self.graph = graph
        self.restarts = restarts
        self.moves = 0
        rng = random.Random(seed)
        best: dict[str, int] = {}
        best_weight = -1.0
        for _ in range(restarts):
            side = {n: rng.randint(0, 1) for n in graph.nodes()}
            self._climb(side)
            weight = self._weight(side)
            if weight > best_weight:
                best, best_weight = side, weight
        self.side = best

    def _gain(self, side: dict[str, int], node: str) -> float:
        # crossing weight gained by flipping node: same-side minus cross-side
        same = 0.0
        cross = 0.0
        for nbr, w in self.graph.neighbors(node).items():
            if side[nbr] == side[node]:
                same += w
            else:
                cross += w
        return same - cross

    def _climb(self, side: dict[str, int]) -> None:
        improved = True
        while improved:
            improved = False
            for node in self.graph.nodes():
                if self._gain(side, node) > 0:
                    side[node] ^= 1
                    self.moves += 1
                    improved = True

    def _weight(self, side: dict[str, int]) -> float:
        return sum(w for u, v, w in self.graph.edges() if side[u] != side[v])

    def cut_weight(self) -> float:
        return self._weight(self.side)

    def gain(self, node: str) -> float:
        return self._gain(self.side, node)

    def sides(self) -> tuple[set[str], set[str]]:
        zero = {n for n, s in self.side.items() if s == 0}
        return zero, set(self.graph.nodes()) - zero

    def total_weight(self) -> float:
        return sum(w for _u, _v, w in self.graph.edges())

    def guarantee_bound(self) -> float:
        return self.total_weight() / 2.0

    def note(self) -> str:
        total = self.total_weight() or 1.0
        return (
            f"cut {self.cut_weight():g} of total {self.total_weight():g} "
            f"({self.cut_weight() / total * 100:.0f}%) over {self.restarts} start(s) and "
            f"{self.moves} move(s); at least half by the local-optimum argument"
        )
