"""Graph entropy: how much surprise a graph holds, measured three ways.

Entropy measures spread. A graph has several distributions to spread
over, and each gives a different reading. The degree entropy is the
Shannon entropy of the degree distribution: a regular graph, where every
node has the same degree, scores zero, and a graph with many distinct
degrees each held by a few nodes scores high. The walk entropy rate is
the long-run bits per step of a random walker: at a node of strength s
with incident weights w the walker chooses with probability w over s,
and the rate weights each node's choice entropy by how often the walker
stands there, which on an undirected graph is the node's strength over
the total. A d-regular graph gives exactly log d bits per step. The
structural entropy of Li and Pan reads the stationary distribution
itself and, given a partition into communities, charges each step twice:
once for picking a node inside its community, once for the community
when the step crosses an edge that leaves it. The trivial partition and
the partition into singletons both score the one-dimensional entropy,
while a partition that follows real community structure scores less,
which is what makes the measure usable for judging a clustering. All
three are in bits. Directed graphs are refused because their stationary
distribution is not a degree ratio, and a partition that does not cover
the nodes exactly once is refused with the missing or repeated node
named.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from math import log2

from mesh.errors import Invalid
from mesh.graph import Graph


class GraphEntropy:
    def __init__(self, graph: Graph) -> None:
        if graph.directed:
            raise Invalid("entropy measures here need an undirected graph")
        self.graph = graph
        self.strength = {
            n: sum(graph.neighbors(n).values()) for n in graph.nodes()
        }
        self.total = sum(self.strength.values())

    def degree_entropy(self) -> float:
        counts = Counter(self.graph.degree(n) for n in self.graph.nodes())
        n = self.graph.node_count()
        if n == 0:
            return 0.0
        # adding 0.0 turns the negative zero of a single degree class into plain zero
        return -sum((c / n) * log2(c / n) for c in counts.values()) + 0.0

    def _stationary(self, node: str) -> float:
        return self.strength[node] / self.total if self.total else 0.0

    def walk_entropy_rate(self) -> float:
        rate = 0.0
        for node in self.graph.nodes():
            s = self.strength[node]
            if s == 0:
                continue
            choice = -sum((w / s) * log2(w / s) for w in self.graph.neighbors(node).values())
            rate += self._stationary(node) * choice
        return rate

    def structural_entropy(self) -> float:
        # the one-dimensional reading: entropy of the stationary distribution
        return (
            -sum(
                self._stationary(n) * log2(self._stationary(n))
                for n in self.graph.nodes()
                if self._stationary(n) > 0
            )
            + 0.0
        )

    def partition_entropy(self, communities: Iterable[Iterable[str]]) -> float:
        parts = [set(c) for c in communities]
        self._check_cover(parts)
        if self.total == 0:
            return 0.0
        total = 0.0
        for part in parts:
            volume = sum(self.strength[n] for n in part)
            if volume == 0:
                continue
            for node in part:
                if self.strength[node] > 0:
                    total -= self._stationary(node) * log2(self.strength[node] / volume)
            leaving = sum(
                w
                for node in part
                for other, w in self.graph.neighbors(node).items()
                if other not in part
            )
            total -= (leaving / self.total) * log2(volume / self.total)
        return total

    def _check_cover(self, parts: list[set[str]]) -> None:
        seen: set[str] = set()
        for part in parts:
            repeated = seen & part
            if repeated:
                raise Invalid(f"node '{min(repeated)}' appears in two communities")
            seen |= part
        missing = set(self.graph.nodes()) - seen
        if missing:
            raise Invalid(f"node '{min(missing)}' is in no community")
        extra = seen - set(self.graph.nodes())
        if extra:
            raise Invalid(f"'{min(extra)}' is not a node of the graph")

    def note(self) -> str:
        return (
            f"degree entropy {self.degree_entropy():.3f} bits, walk entropy rate "
            f"{self.walk_entropy_rate():.3f} bits per step, structural entropy "
            f"{self.structural_entropy():.3f} bits"
        )
