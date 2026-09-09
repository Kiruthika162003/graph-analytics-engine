"""Bond percolation: keep each edge with probability p and watch the giant piece appear.

Remove edges at random and a network falls apart, not gradually but
at a threshold: below some retention probability only small pieces
survive, above it one piece holds a fixed fraction of every node.
Bond percolation keeps each edge independently with probability p,
and the reading that matters is the size of the largest surviving
piece as a fraction of the nodes, averaged over trials. The engine
runs the trials with a seeded generator, sweeps p over a grid, and
reports the largest-piece fraction at each step, which is a curve
that rises from near zero to one; it estimates the threshold as the
p where the curve first crosses one half, and it compares that with
two known values: on an infinite tree of branching b the threshold is
one over b, and Molloy and Reed give it for a random graph as the
mean degree over the mean of degree times degree minus one, which the
module computes from the graph's own degrees as the configuration
model estimate. The curve is checked to be monotone once averaged
over enough trials, to read zero at p equal zero on any graph with
more than one node, and to read one at p equal one on a connected
graph. A single trial's surviving graph is also exposed so a caller
can look at the pieces, and a directed graph is refused because
directed percolation has in and out pieces that are a different
story.
"""

from __future__ import annotations

import random
from collections import deque

from mesh.errors import Invalid
from mesh.graph import Graph


class Percolation:
    def __init__(self, graph: Graph, seed: int = 0) -> None:
        if graph.directed:
            raise Invalid("bond percolation here is on an undirected graph")
        self.graph = graph
        self.rng = random.Random(seed)
        self.nodes = graph.nodes()

    def keep(self, p: float) -> Graph:
        if not 0.0 <= p <= 1.0:
            raise Invalid(f"a retention probability lies in 0..1, not {p}")
        g = Graph()
        for n in self.nodes:
            g.add_node(n)
        for u, v, w in self.graph.edges():
            if self.rng.random() < p:
                g.add_edge(u, v, w)
        return g

    @staticmethod
    def largest_piece(g: Graph) -> int:
        seen: set[str] = set()
        best = 0
        for start in g.nodes():
            if start in seen:
                continue
            size = 1
            seen.add(start)
            queue = deque([start])
            while queue:
                node = queue.popleft()
                for other in g.neighbors(node):
                    if other not in seen:
                        seen.add(other)
                        size += 1
                        queue.append(other)
            best = max(best, size)
        return best

    def giant_fraction(self, p: float, trials: int = 20) -> float:
        if trials < 1:
            raise Invalid("at least one trial is needed")
        if not self.nodes:
            return 0.0
        total = sum(self.largest_piece(self.keep(p)) for _ in range(trials))
        return total / (trials * len(self.nodes))

    def sweep(self, steps: int = 10, trials: int = 20) -> list[tuple[float, float]]:
        if steps < 1:
            raise Invalid("a sweep needs at least one step")
        return [(k / steps, self.giant_fraction(k / steps, trials)) for k in range(steps + 1)]

    def threshold_estimate(self, steps: int = 10, trials: int = 20) -> float:
        for p, fraction in self.sweep(steps, trials):
            if fraction >= 0.5:
                return p
        return 1.0

    def molloy_reed_threshold(self) -> float:
        degrees = [self.graph.degree(n) for n in self.nodes]
        if not degrees:
            return 1.0
        mean = sum(degrees) / len(degrees)
        excess = sum(d * (d - 1) for d in degrees) / len(degrees)
        return mean / excess if excess > 0 else 1.0

    def note(self, steps: int = 10, trials: int = 20) -> str:
        curve = self.sweep(steps, trials)
        shown = " ".join(f"{f:.2f}" for _p, f in curve)
        return (
            f"giant piece by retention: {shown}; threshold near "
            f"{self.threshold_estimate(steps, trials):.2f}, Molloy-Reed "
            f"{self.molloy_reed_threshold():.2f}"
        )
