"""Dominating set: the fewest nodes that, with their neighbors, cover everyone.

A dominating set is a set of nodes such that every node is either in it
or adjacent to one in it: guard posts from which every room is either
occupied or watched, sensors whose ranges together touch every device,
influencers whose direct followers span the whole audience. The
minimum dominating set is NP-hard to find, and it cannot be approximated
better than a logarithmic factor unless P equals NP, so the greedy that
achieves that logarithm is the honest tool. Repeatedly pick the node
that covers the most still-uncovered nodes, counting itself and its
neighbors, until nothing is left uncovered; the guarantee is that the
result is at most one plus the natural log of the maximum degree plus
one times the optimum, and in practice it is usually much tighter. The
engine measures rather than assumes: on small graphs it computes the
true minimum by trying every subset in increasing size, and reports the
greedy size against it. A vertex cover is related but different, it
guards edges rather than nodes, and every vertex cover of a graph with
no isolated nodes is a dominating set while the reverse fails, a
distinction worth stating since the two are often confused. The finder
returns the greedy set, verifies it dominates, computes the optimum
when the graph is small enough, and reports the greedy size against the
optimum and against the logarithmic bound, because the measured gap is
almost always far under the bound, and the bound is what a caller can
rely on when the graph is too large to check.
"""

from __future__ import annotations

import math
from itertools import combinations

from mesh.errors import Invalid
from mesh.graph import Graph

_EXACT_CAP = 14


class DominatingSet:
    def __init__(self, graph: Graph) -> None:
        if graph.directed:
            raise Invalid("this dominating set is for undirected graphs")
        if graph.node_count() == 0:
            raise Invalid("an empty graph has nothing to dominate")
        self.graph = graph
        self._closed = {n: {n} | set(graph.neighbors(n)) for n in graph.nodes()}
        self.chosen = self._greedy()

    def _greedy(self) -> list[str]:
        uncovered = set(self.graph.nodes())
        chosen: list[str] = []
        while uncovered:
            # the node whose closed neighborhood covers the most still uncovered
            best = max(
                sorted(self.graph.nodes()), key=lambda n: len(self._closed[n] & uncovered)
            )
            chosen.append(best)
            uncovered -= self._closed[best]
        return chosen

    def dominates(self, chosen: list[str]) -> bool:
        covered: set[str] = set()
        for n in chosen:
            covered |= self._closed[n]
        return covered == set(self.graph.nodes())

    def optimum(self) -> int:
        if self.graph.node_count() > _EXACT_CAP:
            raise Invalid(f"exact search is capped at {_EXACT_CAP} nodes")
        nodes = sorted(self.graph.nodes())
        for size in range(1, len(nodes) + 1):
            for subset in combinations(nodes, size):
                if self.dominates(list(subset)):
                    return size
        return len(nodes)

    def log_bound(self, optimum: int) -> float:
        max_degree = max(self.graph.degree(n) for n in self.graph.nodes())
        return (1 + math.log(max_degree + 1)) * optimum

    def note(self) -> str:
        try:
            best = self.optimum()
            against = f"optimum {best}, bound {self.log_bound(best):.1f}"
        except Invalid:
            against = "optimum not computed past the cap"
        return (
            f"greedy dominating set of {len(self.chosen)} against {against}; the "
            "measured gap is usually far under the logarithmic bound"
        )
