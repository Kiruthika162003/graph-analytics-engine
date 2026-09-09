"""Degree-preserving edge swaps: shuffle who is joined to whom while every degree stays put.

To ask whether a network's clustering or assortativity is remarkable,
compare it with graphs that have the same degrees and nothing else in
common, and the standard way to draw those is the double edge swap.
Pick two edges a-b and c-d, and replace them with a-d and c-b. Every
degree is unchanged, since each node loses one edge and gains one, and
the swap is refused when it would create a self loop, because a equals
d or c equals b, or a repeated edge, because a-d or c-b already exists.
Repeating the swap many times walks the space of graphs with that
degree sequence, and the walk is a Markov chain whose stationary
distribution is uniform over simple graphs with the sequence when
refused swaps are counted as stays. The engine performs a requested
number of attempts with a seeded generator, counts the accepted ones,
returns a new graph rather than altering the input, and verifies the
degree sequence and edge count are preserved and the result is simple.
A graph with fewer than two edges has nothing to swap and is returned
as it is with zero accepted; a directed graph is refused, and a swap
count below zero is refused. The note gives the acceptance rate, which
is itself a reading: a dense graph refuses most swaps because the new
edges already exist, and a sparse one accepts nearly all.
"""

from __future__ import annotations

import random

from mesh.errors import Invalid
from mesh.graph import Graph


class EdgeSwap:
    def __init__(self, graph: Graph, seed: int = 0) -> None:
        if graph.directed:
            raise Invalid("degree-preserving swaps are defined here on undirected graphs")
        self.graph = graph
        self.rng = random.Random(seed)
        self.attempted = 0
        self.accepted = 0

    def shuffle(self, swaps: int) -> Graph:
        if swaps < 0:
            raise Invalid("the swap count cannot be negative")
        edges = [(u, v) for u, v, _w in self.graph.edges()]
        # weights are keyed by the unordered pair, since a swap may flip c and d
        weights = {frozenset((u, v)): w for u, v, w in self.graph.edges()}
        present = set(weights)
        if len(edges) >= 2:
            for _ in range(swaps):
                self.attempted += 1
                i, j = self.rng.sample(range(len(edges)), 2)
                a, b = edges[i]
                c, d = edges[j]
                if self.rng.random() < 0.5:
                    c, d = d, c
                if a == d or c == b:
                    continue
                if frozenset((a, d)) in present or frozenset((c, b)) in present:
                    continue
                present.discard(frozenset((a, b)))
                present.discard(frozenset((c, d)))
                present.add(frozenset((a, d)))
                present.add(frozenset((c, b)))
                w_ab = weights.pop(frozenset((a, b)))
                w_cd = weights.pop(frozenset((c, d)))
                edges[i], edges[j] = (a, d), (c, b)
                weights[frozenset((a, d))] = w_ab
                weights[frozenset((c, b))] = w_cd
                self.accepted += 1
        out = Graph()
        for n in self.graph.nodes():
            out.add_node(n)
        for u, v in edges:
            out.add_edge(u, v, weights[frozenset((u, v))])
        return out

    def degrees_preserved(self, other: Graph) -> bool:
        return all(self.graph.degree(n) == other.degree(n) for n in self.graph.nodes())

    @staticmethod
    def is_simple(other: Graph) -> bool:
        seen: set[frozenset[str]] = set()
        for u, v, _w in other.edges():
            if u == v or frozenset((u, v)) in seen:
                return False
            seen.add(frozenset((u, v)))
        return True

    def edges_changed(self, other: Graph) -> int:
        before = {frozenset((u, v)) for u, v, _w in self.graph.edges()}
        after = {frozenset((u, v)) for u, v, _w in other.edges()}
        return len(after - before)

    def note(self) -> str:
        rate = self.accepted / self.attempted if self.attempted else 0.0
        return (
            f"{self.accepted} of {self.attempted} swap(s) accepted ({rate:.0%}); a low rate "
            "means the new edges kept landing on existing ones, the mark of a dense graph"
        )
