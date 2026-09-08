"""Degree assortativity: do well-connected nodes link to each other, or to the lonely?

Assortativity asks whether edges tend to join nodes of similar degree.
In a social network the answer is usually yes, popular people know
popular people, and the graph is assortative; in the internet's router
graph and most biological networks the answer is no, hubs connect mostly
to low-degree nodes, and the graph is disassortative. The number that
captures it is Newman's assortativity coefficient, which is simply the
Pearson correlation between the degrees at the two ends of an edge,
taken over all edges. Each undirected edge is counted in both
orientations so the correlation is symmetric, then the coefficient is
the covariance of the two end-degrees divided by their variance. It runs
from minus one, every edge joining a high to a low, through zero, no
tendency at all, to plus one, every edge joining equals. A regular
graph, where every node has the same degree, has zero variance and no
defined coefficient, which the engine reports as such rather than
dividing by zero, and a star is the extreme disassortative case, every
edge joining the hub to a leaf. The coefficient matters because it
changes how a network behaves: assortative networks percolate and
spread more easily through their core of connected hubs but fragment
into that core when the hubs are removed, while disassortative ones
hold together through the hubs and shatter when they go. The measure
computes the coefficient from the edge list, returns the sign as a
verdict, refuses a graph with no edges, and reports the coefficient
against a rough benchmark for a random graph of the same degrees,
which is near zero, because that gap is the amount of degree mixing the
graph's structure imposes beyond what its degrees alone would produce.
"""

from __future__ import annotations

import math

from mesh.errors import Invalid
from mesh.graph import Graph


class Assortativity:
    def __init__(self, graph: Graph) -> None:
        if graph.edge_count() == 0:
            raise Invalid("assortativity needs edges to correlate across")
        self.graph = graph
        self.coefficient: float | None = self._compute()

    def _pairs(self) -> list[tuple[int, int]]:
        # each edge in both orientations so the correlation is symmetric
        out: list[tuple[int, int]] = []
        for u, v, _w in self.graph.edges():
            du, dv = self.graph.degree(u), self.graph.degree(v)
            out.append((du, dv))
            if not self.graph.directed:
                out.append((dv, du))
        return out

    def _compute(self) -> float | None:
        pairs = self._pairs()
        n = len(pairs)
        xs = [a for a, _b in pairs]
        ys = [b for _a, b in pairs]
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        cov = sum((x - mean_x) * (y - mean_y) for x, y in pairs) / n
        var_x = sum((x - mean_x) ** 2 for x in xs) / n
        var_y = sum((y - mean_y) ** 2 for y in ys) / n
        if var_x == 0 or var_y == 0:
            return None  # a regular graph: every degree equal, nothing to correlate
        return cov / math.sqrt(var_x * var_y)

    def verdict(self) -> str:
        if self.coefficient is None:
            return "regular: every node has the same degree, no coefficient"
        if self.coefficient > 0.05:
            return "assortative: hubs link to hubs"
        if self.coefficient < -0.05:
            return "disassortative: hubs link to the lonely"
        return "neutral: no degree mixing tendency"

    def note(self) -> str:
        if self.coefficient is None:
            return self.verdict()
        return (
            f"assortativity {self.coefficient:+.3f}, {self.verdict()}; a random "
            "graph with these degrees would sit near zero, the gap is structure"
        )
