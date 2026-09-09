"""Null model comparison: is a graph's clustering more than its degrees alone would give.

A high clustering coefficient means little on its own, because a
graph with many high-degree nodes clusters by accident. The honest
question is whether the observed graph clusters more than a random
graph with the same degree sequence, and the configuration model
answers it: draw many degree-preserving shuffles by double edge
swaps, measure each, and report where the observed value falls. The
engine measures transitivity, three times the triangles over the
connected triples, and assortativity as the Pearson correlation of
degrees across edges, on the observed graph and on a sample of
shuffles, and reports the sample mean, the sample standard deviation,
and the z-score of the observed value, with the fraction of samples
at or above it as an empirical p-value that needs no distributional
assumption. A z-score above two on transitivity is the usual reading
of a network that clusters for a reason, and a lattice-like graph
such as a ring of cliques scores far above that while a random graph
with the same degrees scores near zero by construction. Every shuffle
runs enough swaps to rewire the graph several times over, the seed is
fixed so a report is reproducible, and the degree sequence of every
sample is checked against the original, which is the invariant the
whole comparison rests on. A directed graph is refused.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import combinations
from math import sqrt

from mesh.edgeswap import EdgeSwap
from mesh.errors import Invalid
from mesh.graph import Graph


def transitivity(graph: Graph) -> float:
    triangles = 0
    triples = 0
    for node in graph.nodes():
        nbrs = list(graph.neighbors(node))
        for a, b in combinations(nbrs, 2):
            triples += 1
            if graph.has_edge(a, b):
                triangles += 1
    # each triangle appears at all three corners, each open triple at its middle only
    return triangles / triples if triples else 0.0


def assortativity(graph: Graph) -> float:
    pairs = [(graph.degree(u), graph.degree(v)) for u, v, _w in graph.edges()]
    if not pairs:
        return 0.0
    xs = [x for x, _y in pairs] + [y for _x, y in pairs]
    ys = [y for _x, y in pairs] + [x for x, _y in pairs]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return 0.0
    return cov / sqrt(var_x * var_y)


class NullModel:
    def __init__(self, graph: Graph, samples: int = 30, seed: int = 0) -> None:
        if graph.directed:
            raise Invalid("the configuration model here shuffles undirected edges")
        if samples < 2:
            raise Invalid("at least two samples are needed for a spread")
        self.graph = graph
        self.samples = samples
        self.seed = seed
        swaps = max(10 * graph.edge_count(), 1)
        self.degrees_preserved = True
        self.shuffles: list[Graph] = []
        for i in range(samples):
            swapper = EdgeSwap(graph, seed=seed + i)
            shuffled = swapper.shuffle(swaps)
            if not swapper.degrees_preserved(shuffled):
                self.degrees_preserved = False
            self.shuffles.append(shuffled)

    def _compare(self, measure: Callable[[Graph], float]) -> dict[str, float]:
        observed = measure(self.graph)
        values = [measure(g) for g in self.shuffles]
        mean = sum(values) / len(values)
        spread = sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
        z = (observed - mean) / spread if spread > 0 else 0.0
        p = sum(1 for v in values if v >= observed) / len(values)
        return {"observed": observed, "mean": mean, "spread": spread, "z": z, "p": p}

    def transitivity(self) -> dict[str, float]:
        return self._compare(transitivity)

    def assortativity(self) -> dict[str, float]:
        return self._compare(assortativity)

    def note(self) -> str:
        t = self.transitivity()
        beyond = t["z"] > 2
        verdict = "clusters beyond its degrees" if beyond else "clusters as its degrees allow"
        return (
            f"transitivity {t['observed']:.3f} against a null mean of {t['mean']:.3f} "
            f"(z {t['z']:.2f}, p {t['p']:.2f}) over {self.samples} shuffle(s): {verdict}"
        )
