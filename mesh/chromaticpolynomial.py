"""Chromatic polynomial: how many proper colorings a graph has, as a polynomial in the colors.

For every graph there is a polynomial P such that P(k) is the number
of ways to color the nodes with k colors so that no edge joins two of
the same color. The polynomial comes from deletion and contraction:
pick an edge e, and the colorings of G minus e split into those where
the ends differ, which are the colorings of G, and those where the
ends agree, which are the colorings of G with e contracted. So P(G)
equals P(G minus e) minus P(G contracted on e), and the recursion
bottoms out at an edgeless graph on n nodes, whose polynomial is k to
the n. The engine runs the recursion on an explicit edge set with a
memo keyed on the graph's shape, keeps the polynomial as integer
coefficients from the constant term up, evaluates it exactly at any
k, and reads the chromatic number as the smallest k with a positive
count. Known closed forms pin the arithmetic: a tree on n nodes gives
k times (k minus 1) to the n minus 1, a cycle gives (k minus 1) to the
n plus (minus 1) to the n times (k minus 1), and a complete graph
gives the falling factorial. Brute-force counting of colorings on
small graphs checks the evaluations directly. The recursion is
exponential and is refused above twelve nodes, which is the honest
limit for exact work here; a directed graph is refused since coloring
ignores direction.
"""

from __future__ import annotations

from functools import cache

from mesh.errors import Invalid
from mesh.graph import Graph

Poly = tuple[int, ...]


def _sub(a: Poly, b: Poly) -> Poly:
    size = max(len(a), len(b))
    out = [0] * size
    for i, c in enumerate(a):
        out[i] += c
    for i, c in enumerate(b):
        out[i] -= c
    return tuple(out)


def _power(n: int) -> Poly:
    return tuple([0] * n + [1])


class ChromaticPolynomial:
    def __init__(self, graph: Graph) -> None:
        if graph.directed:
            raise Invalid("coloring ignores direction; pass an undirected graph")
        if graph.node_count() > 12:
            raise Invalid("deletion-contraction is exponential; keep it to twelve nodes")
        self.graph = graph
        self.calls = 0
        nodes = tuple(graph.nodes())
        edges = frozenset(frozenset((u, v)) for u, v, _w in graph.edges())
        self.coefficients = self._solve(nodes, edges)

    def _solve(self, nodes: tuple[str, ...], edges: frozenset[frozenset[str]]) -> Poly:
        @cache
        def rec(ns: tuple[str, ...], es: frozenset[frozenset[str]]) -> Poly:
            self.calls += 1
            if not es:
                return _power(len(ns))
            edge = min(es, key=sorted)
            u, v = sorted(edge)
            deleted = es - {edge}
            # contract v into u: every edge at v moves to u, and the merged edge vanishes
            merged: set[frozenset[str]] = set()
            for e in deleted:
                if v in e:
                    other = next(iter(e - {v}))
                    if other != u:
                        merged.add(frozenset((u, other)))
                else:
                    merged.add(e)
            contracted_nodes = tuple(n for n in ns if n != v)
            return _sub(rec(ns, deleted), rec(contracted_nodes, frozenset(merged)))

        return rec(nodes, edges)

    def evaluate(self, k: int) -> int:
        return sum(c * k**i for i, c in enumerate(self.coefficients))

    def chromatic_number(self) -> int:
        k = 0
        while self.evaluate(k) <= 0:
            k += 1
            if k > self.graph.node_count() + 1:
                raise Invalid("no color count gives a positive count, which cannot happen")
        return k

    def degree(self) -> int:
        return len(self.coefficients) - 1

    def leading_terms_hold(self) -> bool:
        # degree n, leading coefficient 1, next coefficient minus the edge count
        n = self.graph.node_count()
        if n == 0:
            return self.coefficients == (1,)
        return (
            self.degree() == n
            and self.coefficients[-1] == 1
            and self.coefficients[-2] == -self.graph.edge_count()
        )

    def note(self) -> str:
        terms = []
        for i in range(self.degree(), -1, -1):
            c = self.coefficients[i]
            if c:
                terms.append(f"{c}k^{i}" if i else f"{c}")
        return (
            f"P(k) = {' + '.join(terms) if terms else '0'}; chromatic number "
            f"{self.chromatic_number()} from {self.calls} recursion call(s)"
        )
