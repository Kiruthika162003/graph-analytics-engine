"""Cover time: how long a random walk takes to visit every node, measured against theory.

A random walk visits nodes in proportion to their degree in the long
run, but a different question is how long it takes to see every node at
least once: the cover time. It is the expected number of steps a
crawler spends before it has touched the whole graph, the time a rumor
passed to a random neighbor needs to reach everyone, and it varies
wildly with shape. Two classical bounds frame it. Any connected graph
has cover time at most twice the edge count times the node count minus
one, from the commute-time argument along a spanning tree, and no graph
can be covered faster than about the node count times its natural log,
the coupon-collector floor that the complete graph achieves. The
lollipop graph, a clique with a long tail, comes close to the cubic
upper bound, while a complete graph sits at the logarithmic floor, so
the two bounds are both tight on the right inputs. The engine measures
rather than derives: it runs many seeded walks from a start node,
records the step at which the last new node appeared, and reports the
mean as the cover time estimate, with the standard error so the reader
knows how much the estimate can be trusted. It checks the estimate
against both bounds, refusing a disconnected graph since its cover time
is infinite, and reports where the estimate sits between floor and
ceiling, because a graph near the ceiling has a bottleneck the walk keeps
failing to cross and a graph near the floor is one where every node is a
short hop from everywhere.
"""

from __future__ import annotations

import math
import random

from mesh.connectedcomponents import ConnectedComponents
from mesh.errors import Invalid, Missing
from mesh.graph import Graph


class CoverTime:
    def __init__(self, graph: Graph, start: str, walks: int = 200, seed: int = 0) -> None:
        if graph.directed:
            raise Invalid("this cover-time estimate is for undirected graphs")
        if not graph.has_node(start):
            raise Missing(f"start '{start}' is not in the graph")
        if walks < 1:
            raise Invalid("at least one walk is needed")
        if not ConnectedComponents(graph).is_connected():
            raise Invalid("a disconnected graph has infinite cover time")
        self.graph = graph
        self.start = start
        self.walks = walks
        rng = random.Random(seed)
        self.samples = [self._one_walk(rng) for _ in range(walks)]

    def _one_walk(self, rng: random.Random) -> int:
        n = self.graph.node_count()
        seen = {self.start}
        node = self.start
        steps = 0
        while len(seen) < n:
            node = rng.choice(list(self.graph.neighbors(node)))
            seen.add(node)
            steps += 1
        return steps

    def estimate(self) -> float:
        return sum(self.samples) / len(self.samples)

    def standard_error(self) -> float:
        mean = self.estimate()
        var = sum((s - mean) ** 2 for s in self.samples) / max(1, len(self.samples) - 1)
        return math.sqrt(var / len(self.samples))

    def upper_bound(self) -> float:
        return 2.0 * self.graph.edge_count() * (self.graph.node_count() - 1)

    def lower_bound(self) -> float:
        n = self.graph.node_count()
        return n * math.log(n) if n > 1 else 0.0

    def within_bounds(self) -> bool:
        return self.estimate() <= self.upper_bound()

    def position(self) -> float:
        # 0 at the logarithmic floor, 1 at the cubic ceiling
        low, high = self.lower_bound(), self.upper_bound()
        return (self.estimate() - low) / (high - low) if high > low else 0.0

    def note(self) -> str:
        return (
            f"cover time about {self.estimate():.0f} steps (se {self.standard_error():.1f}) "
            f"from '{self.start}', between a floor of {self.lower_bound():.0f} and a ceiling "
            f"of {self.upper_bound():.0f}; near the ceiling is a bottleneck the walk keeps "
            "failing to cross"
        )
