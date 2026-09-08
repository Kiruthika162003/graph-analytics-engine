"""Euler tour: flatten a tree so every subtree is one contiguous range.

Walk a rooted tree depth-first and write down each node when you enter
it and when you leave it. The entry positions alone give a sequence in
which every subtree occupies a contiguous block, from its root's entry
to the last entry of its deepest descendant, so a question about a
whole subtree, the maximum value in it, how many nodes it holds, whether
a node lies inside it, becomes a question about one array range. That
is the Euler tour technique, and it is why subtree queries on a tree
cost the same as range queries on an array. Ancestry falls out for free:
u is an ancestor of v exactly when v's entry time lies inside u's
interval, entry at or after u's entry and exit at or before u's exit, the
parenthesis nesting that depth-first timestamps always produce. The
engine records entry and exit times with an iterative traversal, keeps
the entry order as the flattened array, builds a segment tree over the
node values in that order so subtree maximum is a single range query,
and answers subtree size as exit minus entry over two plus one, since
each node is written twice. A point update on a node touches one array
position and every subtree containing it sees the change through the
range structure without any tree walking. This complements heavy-light
decomposition, which flattens root-to-node paths into few ranges, by
flattening subtrees into exactly one; a query along a path wants the
first, a query over a subtree wants this. The tour reports the entry
and exit of a node, the subtree size and maximum, whether one node is
an ancestor of another, and it reports the array length against the
node count, because the tour writes each node exactly twice and a length
that is not twice the count is a traversal that skipped or repeated.
"""

from __future__ import annotations

from mesh.errors import Invalid, Missing
from mesh.graph import Graph
from mesh.segmenttree import SegmentTree


class EulerTour:
    def __init__(self, graph: Graph, root: str, values: dict[str, float]) -> None:
        if graph.directed or graph.edge_count() != graph.node_count() - 1:
            raise Invalid("the input must be an undirected tree")
        if not graph.has_node(root):
            raise Missing(f"root '{root}' is not in the graph")
        self.graph = graph
        self.root = root
        self.entry: dict[str, int] = {}
        self.exit: dict[str, int] = {}
        self.order: list[str] = []
        self.tour: list[str] = []
        self._walk()
        if len(self.order) != graph.node_count():
            raise Invalid("the graph is not connected, so it is not a tree")
        self.tree = SegmentTree(size=len(self.order))
        for node, value in values.items():
            self.tree.update(self.entry[node], value)

    def _walk(self) -> None:
        # entry on push, exit when the node's children are exhausted
        stack: list[tuple[str, str | None, list[str]]] = [
            (self.root, None, sorted(self.graph.neighbors(self.root)))
        ]
        self.entry[self.root] = 0
        self.order.append(self.root)
        self.tour.append(self.root)
        while stack:
            node, parent, pending = stack[-1]
            if pending:
                child = pending.pop()
                if child == parent:
                    continue
                self.entry[child] = len(self.order)
                self.order.append(child)
                self.tour.append(child)
                stack.append((child, node, sorted(self.graph.neighbors(child))))
            else:
                self.exit[node] = len(self.order) - 1
                self.tour.append(node)
                stack.pop()

    def is_ancestor(self, u: str, v: str) -> bool:
        for n in (u, v):
            if n not in self.entry:
                raise Missing(f"node '{n}' is not in the tree")
        return self.entry[u] <= self.entry[v] and self.exit[v] <= self.exit[u]

    def subtree_size(self, node: str) -> int:
        if node not in self.entry:
            raise Missing(f"node '{node}' is not in the tree")
        return self.exit[node] - self.entry[node] + 1

    def subtree_max(self, node: str) -> float:
        if node not in self.entry:
            raise Missing(f"node '{node}' is not in the tree")
        return self.tree.range_max(self.entry[node], self.exit[node])

    def update(self, node: str, value: float) -> None:
        if node not in self.entry:
            raise Missing(f"node '{node}' is not in the tree")
        self.tree.update(self.entry[node], value)

    def note(self) -> str:
        return (
            f"tour of length {len(self.tour)} over {self.graph.node_count()} node(s), "
            "each written on entry and exit; a length that is not twice the count is a "
            "traversal that skipped or repeated"
        )
