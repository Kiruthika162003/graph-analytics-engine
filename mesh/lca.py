"""Lowest common ancestor: jump up a tree in powers of two to meet in the middle.

The lowest common ancestor of two nodes in a rooted tree is the deepest
node that is an ancestor of both: the point where their paths to the root
first merge. It answers where two files' directories diverge, which
manager two employees first share, and it is the key to tree distance,
since the distance between two nodes is the sum of their depths minus twice
the depth of their lowest common ancestor. Walking both nodes up one step
at a time until they meet works but costs the tree's depth per query, and
a deep tree with many queries makes that a scan each time. Binary lifting
answers each query in a logarithm of the depth after a one-time
preparation. For every node it records its ancestor one step up, two steps
up, four, eight, and so on, each entry computed from the previous power's
entry of the previous power's ancestor, so the whole table fills in nodes
times log depth. A query first lifts the deeper node up by exactly the
depth difference, decomposing that difference into powers of two and
taking the matching jumps, so both nodes sit at the same depth. Then, if
they are not already the same node, it walks the powers from largest to
smallest and jumps both nodes up by a power whenever their ancestors at
that power differ, which keeps them just below the meeting point; after the
walk their common parent is the answer. The preparation needs the tree
rooted, which a breadth-first search from the chosen root provides along
with each node's depth. The finder answers the lowest common ancestor and
the tree distance between any two nodes, refuses a node outside the tree
and a graph that is not a tree, and reports the table depth, the number of
power-of-two levels, because that is the logarithm the query cost is
measured in.
"""

from __future__ import annotations

from mesh.bfs import BFS
from mesh.errors import Invalid, Missing
from mesh.graph import Graph


class LowestCommonAncestor:
    def __init__(self, graph: Graph, root: str) -> None:
        if graph.directed:
            raise Invalid("the tree is given as an undirected graph rooted at a node")
        if graph.edge_count() != graph.node_count() - 1:
            raise Invalid("a tree has exactly node-count-minus-one edges")
        search = BFS(graph, root)
        if len(search.distance) != graph.node_count():
            raise Invalid("the graph is not connected, so it is not a tree")
        self.graph = graph
        self.root = root
        self.depth = search.distance
        self.levels = max(1, (graph.node_count()).bit_length())
        # up[k][v] is the ancestor of v exactly 2**k steps above it
        self.up: list[dict[str, str]] = [{} for _ in range(self.levels)]
        for node, parent in search.parent.items():
            self.up[0][node] = parent if parent is not None else node
        for k in range(1, self.levels):
            prev = self.up[k - 1]
            self.up[k] = {v: prev[prev[v]] for v in prev}

    def _lift(self, node: str, steps: int) -> str:
        # decompose the step count into powers of two
        k = 0
        while steps:
            if steps & 1:
                node = self.up[k][node]
            steps >>= 1
            k += 1
        return node

    def lca(self, a: str, b: str) -> str:
        for n in (a, b):
            if n not in self.depth:
                raise Missing(f"'{n}' is not in the tree")
        if self.depth[a] < self.depth[b]:
            a, b = b, a
        a = self._lift(a, self.depth[a] - self.depth[b])
        if a == b:
            return a
        for k in range(self.levels - 1, -1, -1):
            if self.up[k][a] != self.up[k][b]:
                a, b = self.up[k][a], self.up[k][b]
        return self.up[0][a]

    def distance(self, a: str, b: str) -> int:
        ancestor = self.lca(a, b)
        return self.depth[a] + self.depth[b] - 2 * self.depth[ancestor]

    def note(self) -> str:
        return (
            f"{self.levels} power-of-two level(s) over {self.graph.node_count()} "
            "node(s); each query lifts in that many jumps at most, the logarithm "
            "the cost is measured in"
        )
