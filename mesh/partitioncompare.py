"""Partition comparison: how alike two groupings of the same nodes are, and how good each is.

Two community methods on one graph give two partitions, and the
reader asks two things: how much they agree, and which is better. For
agreement, three standard readings live here. The Rand index counts
node pairs that both partitions treat the same way, together or
apart, over all pairs. The Jaccard index counts pairs together in
both over pairs together in either. Normalised mutual information
reads the two as random variables over the nodes and reports their
shared entropy scaled by the mean of their entropies, which is one
for identical partitions up to relabeling and zero when knowing one
says nothing about the other. For quality, modularity is the
fraction of edges inside groups minus the fraction expected from the
degrees alone, and coverage is the fraction of edges inside groups
on its own. The identities the tests hold are the ones that make
these numbers trustworthy: identical partitions score one on every
agreement reading whatever the group labels, the partition into
singletons and the partition into one group score zero mutual
information against each other, all three agreement readings are
symmetric, two cliques on a bridge have positive modularity under
their natural split and negative under a split that halves each
clique, and coverage is one when nothing crosses. A partition that
misses or repeats a node is refused by name.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from math import log

from mesh.errors import Invalid
from mesh.graph import Graph

Partition = list[list[str]]


def _labels(nodes: list[str], partition: Partition) -> dict[str, int]:
    labels: dict[str, int] = {}
    for i, group in enumerate(partition):
        for n in group:
            if n in labels:
                raise Invalid(f"node '{n}' appears in two groups")
            labels[n] = i
    missing = [n for n in nodes if n not in labels]
    if missing:
        raise Invalid(f"node '{missing[0]}' is in no group")
    extra = [n for n in labels if n not in set(nodes)]
    if extra:
        raise Invalid(f"'{extra[0]}' is not a node of the graph")
    return labels


class PartitionCompare:
    def __init__(self, graph: Graph, first: Partition, second: Partition) -> None:
        self.graph = graph
        self.nodes = graph.nodes()
        self.a = _labels(self.nodes, first)
        self.b = _labels(self.nodes, second)

    def _pair_counts(self) -> tuple[int, int, int, int]:
        both = neither = only_a = only_b = 0
        for x, y in combinations(self.nodes, 2):
            same_a = self.a[x] == self.a[y]
            same_b = self.b[x] == self.b[y]
            if same_a and same_b:
                both += 1
            elif not same_a and not same_b:
                neither += 1
            elif same_a:
                only_a += 1
            else:
                only_b += 1
        return both, neither, only_a, only_b

    def rand(self) -> float:
        both, neither, only_a, only_b = self._pair_counts()
        total = both + neither + only_a + only_b
        return (both + neither) / total if total else 1.0

    def jaccard(self) -> float:
        both, _neither, only_a, only_b = self._pair_counts()
        union = both + only_a + only_b
        return both / union if union else 1.0

    @staticmethod
    def _entropy(labels: dict[str, int]) -> float:
        n = len(labels)
        return -sum(c / n * log(c / n) for c in Counter(labels.values()).values()) if n else 0.0

    def nmi(self) -> float:
        n = len(self.nodes)
        if n == 0:
            return 1.0
        joint = Counter((self.a[x], self.b[x]) for x in self.nodes)
        count_a = Counter(self.a.values())
        count_b = Counter(self.b.values())
        mutual = 0.0
        for (i, j), c in joint.items():
            mutual += c / n * log((c * n) / (count_a[i] * count_b[j]))
        ha, hb = self._entropy(self.a), self._entropy(self.b)
        if ha + hb == 0:
            return 1.0
        return mutual / ((ha + hb) / 2)

    def modularity(self, which: str = "first") -> float:
        labels = self.a if which == "first" else self.b
        m = self.graph.edge_count()
        if m == 0:
            return 0.0
        inside = sum(1 for u, v, _w in self.graph.edges() if labels[u] == labels[v])
        volume: dict[int, int] = Counter()
        for n in self.nodes:
            volume[labels[n]] += self.graph.degree(n)
        expected = sum(v * v for v in volume.values()) / (4 * m * m)
        return inside / m - expected

    def coverage(self, which: str = "first") -> float:
        labels = self.a if which == "first" else self.b
        m = self.graph.edge_count()
        if m == 0:
            return 1.0
        return sum(1 for u, v, _w in self.graph.edges() if labels[u] == labels[v]) / m

    def note(self) -> str:
        return (
            f"rand {self.rand():.3f}, jaccard {self.jaccard():.3f}, nmi {self.nmi():.3f}; "
            f"modularity {self.modularity('first'):.3f} against {self.modularity('second'):.3f}"
        )
