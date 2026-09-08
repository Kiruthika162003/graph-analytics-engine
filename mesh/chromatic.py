"""Chromatic number: the true minimum colors, found by search and bracketed by bounds.

The greedy colorers give a proper coloring and admit it may not be
optimal. This module finds the optimum, the chromatic number, for graphs
small enough to search, and says how far the greedy answer was from it.
The search tries k colors for increasing k, and for each k backtracks:
assign the next node the smallest color no neighbor already holds, and
when every color is blocked, back up and change an earlier choice. Two
pieces of pruning keep the search from being hopeless. The nodes are
ordered by the DSatur rule, most-constrained first, so dead ends appear
early, and a new color is only ever tried if it is at most one more than
the largest color used so far, since the colors are interchangeable and
trying color five before color four would only rediscover the same
colorings under a renaming. The lower bound is the largest clique the
Bron-Kerbosch enumerator finds, because a clique of size k needs k
colors, and the upper bound is the DSatur greedy result; the search runs
only for k between them, and when the two bounds meet the answer is
known with no search at all. The cost is still exponential in the worst
case and the module caps the node count it will attempt, refusing
larger graphs rather than running for hours, which is the honest shape
of an exact method for an NP-hard problem. The solver returns the
chromatic number, a coloring that achieves it, both bounds, and reports
the greedy answer against the optimum, because the gap between them is
what the greedy heuristics leave on the table and, on most sparse
graphs, it is zero.
"""

from __future__ import annotations

from mesh.bronkerbosch import BronKerbosch
from mesh.coloring import Coloring
from mesh.errors import Invalid
from mesh.graph import Graph

_NODE_CAP = 40


class ChromaticNumber:
    def __init__(self, graph: Graph) -> None:
        if graph.directed:
            raise Invalid("coloring is defined on an undirected graph")
        if graph.node_count() == 0:
            raise Invalid("an empty graph has no chromatic number")
        if graph.node_count() > _NODE_CAP:
            raise Invalid(f"more than {_NODE_CAP} nodes; the exact search is capped")
        self.graph = graph
        self._nbrs = {n: set(graph.neighbors(n)) for n in graph.nodes()}
        self.lower = BronKerbosch(graph).clique_number()
        greedy = Coloring(graph, strategy="dsatur")
        self.upper = greedy.color_count()
        self.greedy_count = self.upper
        self.order = sorted(graph.nodes(), key=lambda n: (-graph.degree(n), n))
        self.coloring: dict[str, int] = dict(greedy.color)
        self.value = self._search()

    def _search(self) -> int:
        for k in range(self.lower, self.upper):
            assignment: dict[str, int] = {}
            if self._try(k, 0, assignment, 0):
                self.coloring = assignment
                return k
        return self.upper  # the greedy coloring was already optimal

    def _try(self, k: int, index: int, assignment: dict[str, int], used: int) -> bool:
        if index == len(self.order):
            return True
        node = self.order[index]
        blocked = {assignment[m] for m in self._nbrs[node] if m in assignment}
        # colors are interchangeable: never skip ahead past the next fresh one
        for color in range(min(k, used + 1)):
            if color in blocked:
                continue
            assignment[node] = color
            if self._try(k, index + 1, assignment, max(used, color + 1)):
                return True
            del assignment[node]
        return False

    def is_proper(self) -> bool:
        return all(self.coloring[u] != self.coloring[v] for u, v, _w in self.graph.edges())

    def greedy_gap(self) -> int:
        return self.greedy_count - self.value

    def note(self) -> str:
        how = "no search needed, the bounds met" if self.lower == self.upper else "by search"
        return (
            f"chromatic number {self.value} {how}, greedy used {self.greedy_count}; "
            f"the gap of {self.greedy_gap()} is what the heuristic left on the table"
        )
