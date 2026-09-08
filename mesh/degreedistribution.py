"""Degree distribution: the histogram that says what kind of network this is.

The first thing to compute on an unfamiliar network is how its degrees
are spread. A distribution that clusters tightly around its mean, with a
variance close to the mean, is what a random graph of independent edges
produces, and it says the network has no hubs. A distribution whose tail
runs far past the mean, with a variance many times the mean, says a few
nodes carry a large share of the edges, and if the tail falls off as a
power of the degree, the network is scale-free in the sense that made
the term famous: no characteristic degree, hubs at every scale. The
engine computes the histogram, the mean, the variance, and the ratio of
variance to mean, which is near one for a Poisson-like random graph and
grows with the heaviness of the tail. For the tail it fits a power-law
exponent by the discrete maximum-likelihood estimator over degrees at
or above a chosen minimum, one plus the count over the sum of log of
degree over the minimum less a half, which is the standard estimator
and far more reliable than fitting a line to a log-log histogram, a
method known to give wrong exponents. The fit is honest about what it
is: an estimate of the exponent assuming a power law holds above the
minimum, not a test that one does. The engine reports the estimated
exponent beside the variance ratio, because a preferential-attachment
graph shows an exponent near three and a large ratio, a random graph
shows a ratio near one and an exponent estimate that means nothing,
and the two numbers together tell which the caller is holding.
"""

from __future__ import annotations

import math
from collections import Counter

from mesh.errors import Invalid
from mesh.graph import Graph


class DegreeDistribution:
    def __init__(self, graph: Graph) -> None:
        if graph.node_count() == 0:
            raise Invalid("an empty graph has no degrees to distribute")
        self.graph = graph
        self.degrees = [graph.degree(n) for n in graph.nodes()]

    def histogram(self) -> dict[int, int]:
        return dict(sorted(Counter(self.degrees).items()))

    def mean(self) -> float:
        return sum(self.degrees) / len(self.degrees)

    def variance(self) -> float:
        m = self.mean()
        return sum((d - m) ** 2 for d in self.degrees) / len(self.degrees)

    def variance_ratio(self) -> float:
        m = self.mean()
        return self.variance() / m if m else 0.0

    def max_degree(self) -> int:
        return max(self.degrees)

    def power_law_exponent(self, minimum: int = 1) -> float:
        # discrete maximum likelihood over degrees at or above the minimum
        tail = [d for d in self.degrees if d >= minimum]
        if len(tail) < 2:
            raise Invalid(f"fewer than two degrees at or above {minimum}; no tail to fit")
        denominator = sum(math.log(d / (minimum - 0.5)) for d in tail)
        if denominator <= 0:
            raise Invalid("every degree in the tail equals the minimum; the fit is undefined")
        return 1.0 + len(tail) / denominator

    def note(self) -> str:
        ratio = self.variance_ratio()
        shape = "hub-heavy" if ratio > 2 else "Poisson-like, no hubs"
        try:
            exponent = f"{self.power_law_exponent(max(1, self.max_degree() // 4)):.2f}"
        except Invalid:
            exponent = "undefined"
        return (
            f"mean degree {self.mean():.2f}, variance over mean {ratio:.2f} ({shape}), "
            f"tail exponent {exponent}; an exponent near three with a large ratio is "
            "preferential attachment"
        )
