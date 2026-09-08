"""Graph diff: what changed between two snapshots of the same network.

A network measured twice is two graphs on overlapping node sets, and the
question is what happened in between: which edges appeared, which
vanished, which nodes joined or left, and whose neighborhood shifted
most. The diff is the edge-set difference, computed on unordered pairs
for an undirected graph and ordered pairs for a directed one, together
with the node-set difference and, for the nodes present in both, the
change in degree. Two summaries sit on top. The Jaccard similarity of
the two edge sets, shared edges over the union, says how much of the
structure persisted, one for identical graphs and zero for disjoint
ones. And the churn, added plus removed over the union, is its
complement, the fraction of the combined structure that moved. The
nodes whose degree changed most are the ones the change was about, and
the engine ranks them by absolute delta so a reader sees the hub that
lost half its links before the leaf that gained one. A diff refuses to
compare a directed graph with an undirected one, since an edge would
mean different things on the two sides and the counts would be
nonsense. The differ returns added and removed edges, joined and left
nodes, per-node degree deltas, the similarity and churn, and reports the
churn beside the count of nodes whose degree changed, because a high
churn concentrated on few nodes is a local rewiring around them while a
high churn spread across most nodes is a network that turned over.
"""

from __future__ import annotations

from mesh.errors import Invalid
from mesh.graph import Graph


class GraphDiff:
    def __init__(self, before: Graph, after: Graph) -> None:
        if before.directed != after.directed:
            raise Invalid("cannot diff a directed graph against an undirected one")
        self.before = before
        self.after = after
        self.directed = before.directed
        old = self._edge_keys(before)
        new = self._edge_keys(after)
        self.added = sorted(new - old, key=sorted)
        self.removed = sorted(old - new, key=sorted)
        self.shared = len(old & new)
        self.union = len(old | new)
        self.joined = sorted(set(after.nodes()) - set(before.nodes()))
        self.left = sorted(set(before.nodes()) - set(after.nodes()))

    def _edge_keys(self, g: Graph) -> set[tuple[str, str] | frozenset[str]]:
        # ordered pairs for a digraph, unordered for an undirected one
        if self.directed:
            return {(u, v) for u, v, _w in g.edges()}
        return {frozenset((u, v)) for u, v, _w in g.edges()}

    def degree_delta(self) -> dict[str, int]:
        common = set(self.before.nodes()) & set(self.after.nodes())
        return {n: self.after.degree(n) - self.before.degree(n) for n in sorted(common)}

    def most_changed(self, k: int = 3) -> list[tuple[str, int]]:
        deltas = self.degree_delta()
        return sorted(deltas.items(), key=lambda kv: (-abs(kv[1]), kv[0]))[:k]

    def similarity(self) -> float:
        return self.shared / self.union if self.union else 1.0

    def churn(self) -> float:
        return (len(self.added) + len(self.removed)) / self.union if self.union else 0.0

    def note(self) -> str:
        changed = sum(1 for d in self.degree_delta().values() if d != 0)
        return (
            f"{len(self.added)} edge(s) added, {len(self.removed)} removed, churn "
            f"{self.churn():.2f}, {changed} node(s) with a changed degree; churn on "
            "few nodes is a local rewiring, spread across most is a turnover"
        )
