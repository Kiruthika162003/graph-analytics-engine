"""Centroid decomposition: split a tree at its balance point, again and again.

The centroid of a tree is a node whose removal leaves every remaining
piece at most half the size of the original, and every tree has one,
found by walking from any node toward the heaviest subtree until no
subtree exceeds half. Removing the centroid and recursing on each piece
builds the centroid tree: the centroid at the root, the centroids of the
pieces as its children, and so on. Because each level at least halves
the piece sizes, the centroid tree has depth at most a logarithm of the
node count no matter how unbalanced the original was, and that is the
whole point. A path of a thousand nodes, depth a thousand, gets a
centroid tree of depth ten. Every path in the original tree passes
through exactly one centroid that is an ancestor of both endpoints in
the centroid tree, the first centroid removed that separated them, so a
problem about all paths decomposes into, for each centroid, the paths
through it, and each node belongs to only a logarithm of those
problems. That is the structure behind counting paths of a given
length, finding the nearest marked node, and answering distance queries
with logarithmic updates. The engine finds each centroid by subtree
sizes computed within the current piece, recursing with an explicit
stack, builds the centroid tree as parent pointers, and verifies the
halving property for every centroid it chose. It reports the centroid
tree's depth against the original tree's depth, because the gap between
them is the imbalance the decomposition removed, and a path-shaped
input showing depth ten against a thousand is the guarantee made
visible.
"""

from __future__ import annotations

import math

from mesh.errors import Invalid, Missing
from mesh.graph import Graph


class CentroidDecomposition:
    def __init__(self, graph: Graph) -> None:
        if graph.directed or graph.edge_count() != graph.node_count() - 1:
            raise Invalid("the input must be an undirected tree")
        if graph.node_count() == 0:
            raise Invalid("an empty graph has no centroid")
        self.graph = graph
        self.centroid_parent: dict[str, str | None] = {}
        self.level: dict[str, int] = {}
        self.piece_size: dict[str, int] = {}
        self._decompose()
        if len(self.centroid_parent) != graph.node_count():
            raise Invalid("the graph is not connected, so it is not a tree")

    def _subtree_sizes(self, root: str, removed: set[str]) -> dict[str, int]:
        # sizes within the current piece, rooted at root, ignoring removed nodes
        order: list[str] = []
        parent: dict[str, str | None] = {root: None}
        stack = [root]
        while stack:
            node = stack.pop()
            order.append(node)
            for nbr in self.graph.neighbors(node):
                if nbr not in removed and nbr != parent[node]:
                    parent[nbr] = node
                    stack.append(nbr)
        size: dict[str, int] = {}
        for node in reversed(order):
            size[node] = 1 + sum(
                size[c] for c in self.graph.neighbors(node)
                if c not in removed and parent.get(c) == node
            )
        return size

    def _find_centroid(self, start: str, removed: set[str]) -> str:
        size = self._subtree_sizes(start, removed)
        total = size[start]
        node = start
        came_from: str | None = None
        while True:
            # step toward any subtree heavier than half, else this is it
            heavy = None
            for nbr in self.graph.neighbors(node):
                if nbr in removed or nbr == came_from:
                    continue
                if size.get(nbr, 0) > total // 2:
                    heavy = nbr
                    break
            if heavy is None:
                self.piece_size[node] = total
                return node
            came_from, node = node, heavy

    def _decompose(self) -> None:
        removed: set[str] = set()
        stack: list[tuple[str, str | None, int]] = [(self.graph.nodes()[0], None, 0)]
        while stack:
            start, parent, level = stack.pop()
            c = self._find_centroid(start, removed)
            self.centroid_parent[c] = parent
            self.level[c] = level
            removed.add(c)
            for nbr in self.graph.neighbors(c):
                if nbr not in removed:
                    stack.append((nbr, c, level + 1))

    def depth(self) -> int:
        return max(self.level.values())

    def original_depth(self) -> int:
        # the height of the tree rooted at its first node, by BFS layers
        root = self.graph.nodes()[0]
        depth = {root: 0}
        queue = [root]
        while queue:
            node = queue.pop(0)
            for nbr in self.graph.neighbors(node):
                if nbr not in depth:
                    depth[nbr] = depth[node] + 1
                    queue.append(nbr)
        return max(depth.values())

    def halving_holds(self) -> bool:
        # each centroid's pieces are at most half of the piece it split
        removed: set[str] = set()
        by_level = sorted(self.level, key=lambda n: (self.level[n], n))
        for c in by_level:
            total = self.piece_size[c]
            removed.add(c)
            for nbr in self.graph.neighbors(c):
                if nbr in removed:
                    continue
                if self._subtree_sizes(nbr, removed)[nbr] > total // 2:
                    return False
        return True

    def ancestor_of(self, node: str) -> list[str]:
        if node not in self.centroid_parent:
            raise Missing(f"node '{node}' is not in the tree")
        chain = [node]
        while self.centroid_parent[chain[-1]] is not None:
            chain.append(self.centroid_parent[chain[-1]])  # type: ignore[arg-type]
        return chain

    def note(self) -> str:
        bound = math.ceil(math.log2(self.graph.node_count() + 1))
        return (
            f"centroid tree depth {self.depth()} against original depth "
            f"{self.original_depth()} and a log bound of {bound}; the gap is the "
            "imbalance the decomposition removed"
        )
