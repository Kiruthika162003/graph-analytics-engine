"""Graph merging: union, intersection, and difference, with a stated rule for clashing weights.

Two snapshots of the same network, or two networks over the same
people, are combined more often than they are analysed alone, and
the combination has choices that must be stated: what happens to a
node in only one graph, and what weight an edge in both should carry.
This module makes the choices explicit. The union holds every node
and edge of either graph, and an edge present in both takes its
weight by a named rule: the sum, the larger, the smaller, or the
first graph's. The intersection holds only nodes in both and edges in
both, with the same weight rules. The difference holds the first
graph's nodes and the edges of the first that are absent from the
second, which is the change from one snapshot to the next seen as a
graph. The symmetric difference holds edges in exactly one, which is
what changed in either direction. Every operation refuses to mix a
directed graph with an undirected one, since an arc and an edge do
not compare, and refuses an unknown weight rule by name. The tests
hold the algebra a reader expects: union with the empty graph is the
graph itself, intersection of a graph with itself is itself, the
difference of a graph from itself is edgeless, the union's edge count
is the two counts minus the intersection's, and the symmetric
difference is the union minus the intersection.
"""

from __future__ import annotations

from collections.abc import Callable

from mesh.errors import Invalid
from mesh.graph import Graph

RULES: dict[str, Callable[[float, float], float]] = {
    "sum": lambda a, b: a + b,
    "max": max,
    "min": min,
    "first": lambda a, _b: a,
}


class GraphMerge:
    def __init__(self, first: Graph, second: Graph, rule: str = "sum") -> None:
        if first.directed != second.directed:
            raise Invalid("cannot merge a directed graph with an undirected one")
        if rule not in RULES:
            raise Invalid(f"no weight rule called '{rule}'; try {', '.join(RULES)}")
        self.first = first
        self.second = second
        self.rule = RULES[rule]
        self.directed = first.directed

    def _key(self, u: str, v: str) -> tuple[str, str]:
        return (u, v) if self.directed else (min(u, v), max(u, v))

    def _edges(self, graph: Graph) -> dict[tuple[str, str], float]:
        return {self._key(u, v): w for u, v, w in graph.edges()}

    def _build(self, nodes: list[str], edges: dict[tuple[str, str], float]) -> Graph:
        g = Graph(directed=self.directed)
        for n in nodes:
            g.add_node(n)
        for (u, v), w in sorted(edges.items()):
            g.add_edge(u, v, w)
        return g

    def union(self) -> Graph:
        nodes = list(self.first.nodes())
        nodes.extend(n for n in self.second.nodes() if n not in set(nodes))
        a, b = self._edges(self.first), self._edges(self.second)
        merged = dict(a)
        for key, w in b.items():
            merged[key] = self.rule(a[key], w) if key in a else w
        return self._build(nodes, merged)

    def intersection(self) -> Graph:
        inside = set(self.second.nodes())
        nodes = [n for n in self.first.nodes() if n in inside]
        a, b = self._edges(self.first), self._edges(self.second)
        shared = {key: self.rule(w, b[key]) for key, w in a.items() if key in b}
        return self._build(nodes, shared)

    def difference(self) -> Graph:
        a, b = self._edges(self.first), self._edges(self.second)
        return self._build(self.first.nodes(), {k: w for k, w in a.items() if k not in b})

    def symmetric_difference(self) -> Graph:
        nodes = list(self.first.nodes())
        nodes.extend(n for n in self.second.nodes() if n not in set(nodes))
        a, b = self._edges(self.first), self._edges(self.second)
        only = {k: w for k, w in a.items() if k not in b}
        only.update({k: w for k, w in b.items() if k not in a})
        return self._build(nodes, only)

    def counts_agree(self) -> bool:
        union = self.union().edge_count()
        inter = self.intersection().edge_count()
        both = self.first.edge_count() + self.second.edge_count()
        symmetric = self.symmetric_difference().edge_count()
        return union == both - inter and symmetric == union - inter

    def note(self) -> str:
        counts = {
            "union": self.union().edge_count(),
            "intersection": self.intersection().edge_count(),
            "difference": self.difference().edge_count(),
            "symmetric": self.symmetric_difference().edge_count(),
        }
        return ", ".join(f"{k} {v}" for k, v in counts.items()) + " edge(s)"
