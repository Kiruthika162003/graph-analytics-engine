"""Graph isomorphism: the same graph under different names, decided by refinement then search.

Two graphs are isomorphic when relabelling the nodes of one produces the
other exactly: the same shape drawn with different names. Deciding it is
famously neither known to be easy nor known to be hard, but two tools
cover practice. The first is Weisfeiler-Lehman color refinement, a fast
necessary test. Every node starts with the same color, then each round a
node's new color is a hash of its old color together with the sorted
multiset of its neighbors' old colors, and rounds continue until the
partition into colors stops changing. Two isomorphic graphs must produce
the same multiset of final colors, so if the multisets differ the graphs
are not isomorphic, and that catches almost every non-isomorphic pair in
a handful of cheap rounds. The test is not sufficient: some pairs of
non-isomorphic graphs, regular graphs above all, refine to identical
colors, which is exactly why the second tool exists. The second is a
backtracking search for an explicit mapping, extending a partial
node-to-node assignment one node at a time and checking at every step
that every edge among the assigned nodes is matched by an edge among
their images and vice versa, pruning any candidate whose refined color
or degree differs. The colors from refinement drive the pruning, so the
search is fast where refinement already nearly decided and only becomes
expensive on the highly symmetric graphs where refinement is blind. The
decider runs refinement, returns early on a mismatch, otherwise searches,
returns the mapping when one exists, and reports which stage decided,
because a decision by refinement cost almost nothing while a decision
by search is the honest price of a symmetric input.
"""

from __future__ import annotations

from collections import Counter

from mesh.errors import Invalid
from mesh.graph import Graph


class Isomorphism:
    def __init__(self, a: Graph, b: Graph, max_rounds: int = 20) -> None:
        if a.directed != b.directed:
            raise Invalid("cannot compare a directed graph with an undirected one")
        self.a = a
        self.b = b
        self.decided_by = "counts"
        self.mapping: dict[str, str] | None = None
        self.isomorphic = self._decide(max_rounds)

    @staticmethod
    def _refine(g: Graph, rounds: int) -> dict[str, int]:
        color = dict.fromkeys(g.nodes(), 0)
        for _ in range(rounds):
            signature = {
                n: (color[n], tuple(sorted(color[m] for m in g.neighbors(n))))
                for n in g.nodes()
            }
            palette = {sig: i for i, sig in enumerate(sorted(set(signature.values())))}
            new = {n: palette[signature[n]] for n in g.nodes()}
            if len(set(new.values())) == len(set(color.values())):
                return new  # the partition stopped splitting
            color = new
        return color

    def _decide(self, rounds: int) -> bool:
        same_nodes = self.a.node_count() == self.b.node_count()
        same_edges = self.a.edge_count() == self.b.edge_count()
        if not (same_nodes and same_edges):
            return False
        self.color_a = self._refine(self.a, rounds)
        self.color_b = self._refine(self.b, rounds)
        self.decided_by = "refinement"
        if Counter(self.color_a.values()) != Counter(self.color_b.values()):
            return False
        self.decided_by = "search"
        mapping: dict[str, str] = {}
        return self._search(sorted(self.a.nodes()), mapping, set())

    def _consistent(self, mapping: dict[str, str], u: str, v: str) -> bool:
        # every edge between assigned nodes must be mirrored, both ways
        for x, y in mapping.items():
            if self.a.has_edge(u, x) != self.b.has_edge(v, y):
                return False
            if self.a.has_edge(x, u) != self.b.has_edge(y, v):
                return False
        return True

    def _search(self, order: list[str], mapping: dict[str, str], used: set[str]) -> bool:
        if len(mapping) == len(order):
            self.mapping = dict(mapping)
            return True
        u = order[len(mapping)]
        for v in sorted(self.b.nodes()):
            if v in used or self.color_a[u] != self.color_b[v]:
                continue
            if self.a.degree(u) != self.b.degree(v):
                continue
            if not self._consistent(mapping, u, v):
                continue
            mapping[u] = v
            used.add(v)
            if self._search(order, mapping, used):
                return True
            del mapping[u]
            used.discard(v)
        return False

    def note(self) -> str:
        verdict = "isomorphic" if self.isomorphic else "not isomorphic"
        return (
            f"{verdict}, decided by {self.decided_by}; refinement is nearly free, "
            "search is the honest price of a symmetric input"
        )
