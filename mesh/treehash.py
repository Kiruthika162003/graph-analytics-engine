"""Tree isomorphism by canonical strings: two trees match when their sorted shapes agree.

Isomorphism is hard in general and easy on trees, because a rooted
tree has a canonical name: a leaf is the empty pair of brackets, and
any other node is the sorted concatenation of its children's names
wrapped in brackets, which is the Aho, Hopcroft, and Ullman scheme.
Two rooted trees are isomorphic exactly when their root names are
equal, since the name records the whole shape and sorting removes the
order of the children. An unrooted tree is rooted at its center, the
one or two nodes left when leaves are peeled in rounds, and when there
are two centers both are tried and the smaller name kept, so any two
unrooted trees that are the same shape produce the same name. The
engine peels to the center, names recursively without recursion, so
a long path does not overflow the stack, compares two trees by name,
and counts distinct shapes in a collection, which is how it verifies
Cayley's count from the other side: the labeled trees on four nodes
collapse to exactly two shapes, the path and the star, and on five
nodes to three. A graph that is not a tree is refused by the edge
count and by a peel that gets stuck, and the module also names each
node's subtree so a caller can find repeated sub-shapes.
"""

from __future__ import annotations

from collections import Counter

from mesh.errors import Invalid
from mesh.graph import Graph


class TreeHash:
    def __init__(self, graph: Graph) -> None:
        if graph.directed:
            raise Invalid("tree hashing takes an undirected tree")
        if graph.node_count() and graph.edge_count() != graph.node_count() - 1:
            raise Invalid("a tree has exactly n minus one edges")
        self.graph = graph
        self.centers = self._centers()
        self.names_by_root: dict[str, dict[str, str]] = {}
        for c in self.centers:
            self.names_by_root[c] = self._name_from(c)
        self.canonical = min((self.names_by_root[c][c] for c in self.centers), default="")

    def _centers(self) -> list[str]:
        degree = {n: self.graph.degree(n) for n in self.graph.nodes()}
        remaining = set(degree)
        while len(remaining) > 2:
            leaves = [n for n in remaining if degree[n] <= 1]
            if not leaves:
                raise Invalid("peeling found no leaf, so this graph holds a cycle")
            for leaf in leaves:
                remaining.discard(leaf)
                for m in self.graph.neighbors(leaf):
                    if m in remaining:
                        degree[m] -= 1
        return sorted(remaining)

    def _name_from(self, root: str) -> dict[str, str]:
        # parents by breadth-first order, then names built from the deepest node up
        parent: dict[str, str | None] = {root: None}
        order = [root]
        i = 0
        while i < len(order):
            node = order[i]
            i += 1
            for m in self.graph.neighbors(node):
                if m not in parent:
                    parent[m] = node
                    order.append(m)
        if len(order) != self.graph.node_count():
            raise Invalid("the tree is disconnected")
        children: dict[str, list[str]] = {n: [] for n in order}
        for node, p in parent.items():
            if p is not None:
                children[p].append(node)
        names: dict[str, str] = {}
        for node in reversed(order):
            names[node] = "(" + "".join(sorted(names[c] for c in children[node])) + ")"
        return names

    def same_shape(self, other: TreeHash) -> bool:
        return self.canonical == other.canonical

    def subtree_names(self) -> dict[str, str]:
        return dict(self.names_by_root[self.centers[0]]) if self.centers else {}

    def repeated_subtrees(self) -> dict[str, int]:
        counts = Counter(self.subtree_names().values())
        return {name: n for name, n in counts.items() if n > 1 and name != "()"}

    @staticmethod
    def distinct_shapes(trees: list[Graph]) -> int:
        return len({TreeHash(t).canonical for t in trees})

    def note(self) -> str:
        repeats = len(self.repeated_subtrees())
        return (
            f"tree of {self.graph.node_count()} rooted at {self.centers}: canonical name of "
            f"length {len(self.canonical)}, {repeats} repeated sub-shape(s)"
        )
