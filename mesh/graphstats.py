"""Degree statistics: the moments, the tail, and the fit that tests a tail for a power law.

The first thing measured on a large network is its degree distribution,
and the first claim made about it is usually that the tail follows a
power law. This module holds the arithmetic for making that claim
carefully. It reports the mean degree, the variance, and the second
moment, whose ratio to the mean is the branching factor that the
percolation and epidemic thresholds depend on. It builds the
complementary cumulative distribution, the fraction of nodes with
degree at least k, which is the curve that is read on log axes. And
it fits a discrete power law exponent to the tail above a chosen
minimum degree by the Clauset, Shalizi, and Newman maximum likelihood
estimate, which is one plus n over the sum of log(k over (kmin minus
one half)), a closed form that needs no optimiser. The fit is honest
about its limits: with fewer than a handful of tail nodes it refuses,
and the note reports how many nodes fed the estimate. Against the
fit, a Poisson comparison gives the variance a random graph with the
same mean would have, so a reader sees at once whether the observed
variance is far above it, which is what a heavy tail means before any
exponent is quoted. Every reading is checked on graphs with known
degrees: a regular graph has zero variance and no tail, a star has
one hub, and a graph built with degrees drawn from a fixed power law
returns an exponent near the one it was built from.
"""

from __future__ import annotations

import random
from collections import Counter
from math import log

from mesh.errors import Invalid
from mesh.graph import Graph


class DegreeStats:
    def __init__(self, graph: Graph) -> None:
        self.graph = graph
        self.degrees = sorted((graph.degree(n) for n in graph.nodes()), reverse=True)
        self.counts = Counter(self.degrees)

    def mean(self) -> float:
        return sum(self.degrees) / len(self.degrees) if self.degrees else 0.0

    def second_moment(self) -> float:
        return sum(d * d for d in self.degrees) / len(self.degrees) if self.degrees else 0.0

    def variance(self) -> float:
        return self.second_moment() - self.mean() ** 2

    def branching(self) -> float:
        # the mean excess degree reached by following a random edge
        mean = self.mean()
        return (self.second_moment() - mean) / mean if mean else 0.0

    def poisson_variance(self) -> float:
        return self.mean()

    def heavy_tailed(self, factor: float = 3.0) -> bool:
        return self.variance() > factor * self.poisson_variance()

    def ccdf(self) -> list[tuple[int, float]]:
        n = len(self.degrees)
        out = []
        for k in sorted(self.counts):
            above = sum(1 for d in self.degrees if d >= k)
            out.append((k, above / n))
        return out

    def power_law_exponent(self, kmin: int) -> float:
        if kmin < 1:
            raise Invalid("the tail starts at a positive degree")
        tail = [d for d in self.degrees if d >= kmin]
        if len(tail) < 5:
            raise Invalid(f"only {len(tail)} node(s) at degree {kmin} or above; too few to fit")
        return 1 + len(tail) / sum(log(d / (kmin - 0.5)) for d in tail)

    def tail_size(self, kmin: int) -> int:
        return sum(1 for d in self.degrees if d >= kmin)

    def note(self, kmin: int | None = None) -> str:
        spread = "heavy" if self.heavy_tailed() else "light"
        base = (
            f"mean degree {self.mean():.2f}, variance {self.variance():.2f} against Poisson "
            f"{self.poisson_variance():.2f}; a {spread} tail, branching {self.branching():.2f}"
        )
        if kmin is None:
            return base
        try:
            gamma = self.power_law_exponent(kmin)
        except Invalid as exc:
            return f"{base}; no exponent: {exc}"
        return f"{base}; exponent {gamma:.2f} from {self.tail_size(kmin)} tail node(s)"


def power_law_graph(n: int, gamma: float, kmin: int, seed: int) -> Graph:
    """Build a graph whose degrees are drawn from a discrete power law, by pairing stubs."""
    rng = random.Random(seed)
    if gamma <= 1 or kmin < 1 or n < 2:
        raise Invalid("a power law needs gamma above one, a positive kmin, and two nodes")
    degrees = []
    for _ in range(n):
        # inverse transform on the continuous law, rounded down to a degree
        u = rng.random()
        degrees.append(int((kmin - 0.5) * (1 - u) ** (-1 / (gamma - 1)) + 0.5))
    if sum(degrees) % 2:
        degrees[0] += 1
    stubs = [i for i, d in enumerate(degrees) for _ in range(d)]
    rng.shuffle(stubs)
    g = Graph()
    for i in range(n):
        g.add_node(f"n{i}")
    for a, b in zip(stubs[::2], stubs[1::2], strict=False):
        if a != b and not g.has_edge(f"n{a}", f"n{b}"):
            g.add_edge(f"n{a}", f"n{b}")
    return g
