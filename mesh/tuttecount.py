"""Spanning tree and forest counts by deletion and contraction, checked against Kirchhoff.

The number of spanning trees of a graph obeys the same recursion as
the chromatic polynomial with the sign flipped: for any edge e that is
not a loop, the spanning trees of G are those that avoid e, which are
the spanning trees of G minus e, plus those that use e, which are the
spanning trees of G with e contracted. A bridge has every spanning
tree using it, so deleting it leaves a disconnected graph with zero
trees and the count is the contraction's alone. Contraction here keeps
parallel edges, unlike the coloring recursion, because two parallel
edges give two different trees. The recursion is exponential, but a
memo on the multigraph's shape keeps small graphs quick, and it is
worth having beside the matrix-tree theorem because the two agree only
when both are right: Kirchhoff's determinant and the recursion share
no code. The same recursion with a different base counts spanning
forests when contraction is replaced by counting rooted forests, but
this module keeps to trees and adds one more reading, the number of
spanning trees through a chosen edge, which is the contraction count
alone and tells how load-bearing an edge is: an edge in every tree is
a bridge, and an edge in none is a loop. A directed graph is refused,
and the module refuses more than twelve nodes.
"""

from __future__ import annotations

from collections import Counter
from functools import cache

from mesh.errors import Invalid
from mesh.graph import Graph
from mesh.kirchhoff import SpanningTreeCount

Multi = tuple[tuple[frozenset[str], int], ...]


class TutteCount:
    def __init__(self, graph: Graph) -> None:
        if graph.directed:
            raise Invalid("spanning trees are counted on undirected graphs")
        if graph.node_count() > 12:
            raise Invalid("deletion-contraction is exponential; keep it to twelve nodes")
        self.graph = graph
        self.calls = 0
        self.trees = self._count(tuple(graph.nodes()), self._multi(graph))

    @staticmethod
    def _multi(graph: Graph) -> Multi:
        counts = Counter(frozenset((u, v)) for u, v, _w in graph.edges())
        return tuple(sorted(counts.items(), key=lambda p: sorted(p[0])))

    def _count(self, nodes: tuple[str, ...], edges: Multi) -> int:
        @cache
        def rec(ns: tuple[str, ...], es: Multi) -> int:
            self.calls += 1
            if len(ns) <= 1:
                return 1
            if not es:
                return 0
            (edge, multiplicity), rest = es[0], es[1:]
            u, v = sorted(edge)
            # avoid every parallel copy, or use one of the m copies and contract v into u.
            # the first version dropped one copy on the avoid side and still multiplied
            # the contraction by m, which counted a triangle's trees as four
            merged: Counter[frozenset[str]] = Counter()
            for e, m in rest:
                if v in e:
                    other = next(iter(e - {v}))
                    if other != u:
                        merged[frozenset((u, other))] += m
                else:
                    merged[e] += m
            contracted = tuple(sorted(merged.items(), key=lambda p: sorted(p[0])))
            smaller = tuple(n for n in ns if n != v)
            return rec(ns, rest) + multiplicity * rec(smaller, contracted)

        return rec(nodes, edges)

    def trees_through(self, u: str, v: str) -> int:
        if not self.graph.has_edge(u, v):
            raise Invalid(f"no edge {u}-{v} to count through")
        edge = frozenset((u, v))
        merged: Counter[frozenset[str]] = Counter()
        for e, m in self._multi(self.graph):
            if e == edge:
                continue
            if v in e:
                other = next(iter(e - {v}))
                if other != u:
                    merged[frozenset((u, other))] += m
            else:
                merged[e] += m
        contracted = tuple(sorted(merged.items(), key=lambda p: sorted(p[0])))
        smaller = tuple(n for n in self.graph.nodes() if n != v)
        return self._count(smaller, contracted)

    def is_bridge(self, u: str, v: str) -> bool:
        return self.trees > 0 and self.trees_through(u, v) == self.trees

    def agrees_with_kirchhoff(self) -> bool:
        return self.trees == SpanningTreeCount(self.graph).count

    def note(self) -> str:
        return f"{self.trees} spanning tree(s) by deletion-contraction in {self.calls} call(s)"
