"""Heavy-light decomposition: cut a tree into few chains so any path is few ranges.

A query along a tree path, the maximum weight on the route between two
nodes, say, can be answered by walking the path, which costs its length,
and a deep tree with many queries makes that a scan each time. Heavy-
light decomposition turns every tree path into a small number of
contiguous array ranges so that a range structure, here a segment tree,
answers each piece in a logarithm. The decomposition names, for every
node, its heavy child, the child with the largest subtree, and calls the
edge to it heavy and every other child edge light. Following heavy
edges downward from any node traces a chain, and the tree splits into
chains that partition its nodes. The key fact is that any root-to-node
path crosses at most a logarithm of light edges, because stepping down a
light edge at least halves the subtree size, so it passes through at
most a logarithm of chains. Laying out nodes so each chain occupies a
contiguous block of positions, with the heavy child placed right after
its parent, makes every chain a range. A path query between two nodes
then climbs: whichever endpoint sits on the chain with the deeper head
queries the range from itself up to that head, jumps to the head's
parent, and repeats until both endpoints share a chain, where one final
range finishes it. Each jump is one light edge, so a query costs a
logarithm of range queries, each a logarithm in the segment tree. The
engine builds the decomposition and the segment tree over node values,
answers path maximum, updates a node's value, and reports the chain
count and the most chains any query crossed, because a decomposition on
a path-shaped tree is a single chain, while on a broom it is many, and
that count is the logarithm the guarantee promises to keep small.
"""

from __future__ import annotations

from mesh.errors import Invalid, Missing
from mesh.graph import Graph
from mesh.segmenttree import SegmentTree


class HeavyLight:
    def __init__(self, graph: Graph, root: str, values: dict[str, float]) -> None:
        if graph.directed or graph.edge_count() != graph.node_count() - 1:
            raise Invalid("the input must be an undirected tree")
        if not graph.has_node(root):
            raise Missing(f"root '{root}' is not in the graph")
        self.graph = graph
        self.root = root
        self.parent: dict[str, str | None] = {root: None}
        self.depth: dict[str, int] = {root: 0}
        self.size: dict[str, int] = {}
        self.heavy: dict[str, str | None] = {}
        self.head: dict[str, str] = {}
        self.pos: dict[str, int] = {}
        self.max_chains_crossed = 0
        self._sizes()
        self._decompose()
        if len(self.pos) != graph.node_count():
            raise Invalid("the graph is not connected, so it is not a tree")
        self.tree = SegmentTree(size=graph.node_count())
        for node, value in values.items():
            self.tree.update(self.pos[node], value)

    def _sizes(self) -> None:
        # iterative DFS: record order, then fill sizes and heavy children bottom-up
        order: list[str] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            order.append(node)
            for nbr in self.graph.neighbors(node):
                if nbr != self.parent[node]:
                    self.parent[nbr] = node
                    self.depth[nbr] = self.depth[node] + 1
                    stack.append(nbr)
        for node in reversed(order):
            children = [c for c in self.graph.neighbors(node) if self.parent.get(c) == node]
            self.size[node] = 1 + sum(self.size[c] for c in children)
            if children:
                self.heavy[node] = max(children, key=lambda c: (self.size[c], c))
            else:
                self.heavy[node] = None

    def _decompose(self) -> None:
        # heavy child placed right after its parent so each chain is a range
        counter = 0
        stack: list[tuple[str, str]] = [(self.root, self.root)]
        while stack:
            node, head = stack.pop()
            chain_node: str | None = node
            while chain_node is not None:
                self.head[chain_node] = head
                self.pos[chain_node] = counter
                counter += 1
                for c in self.graph.neighbors(chain_node):
                    if self.parent.get(c) == chain_node and c != self.heavy[chain_node]:
                        stack.append((c, c))  # a light child starts its own chain
                chain_node = self.heavy[chain_node]

    def path_max(self, a: str, b: str) -> float:
        for n in (a, b):
            if n not in self.pos:
                raise Missing(f"node '{n}' is not in the tree")
        best = float("-inf")
        crossed = 0
        while self.head[a] != self.head[b]:
            if self.depth[self.head[a]] < self.depth[self.head[b]]:
                a, b = b, a
            head = self.head[a]
            best = max(best, self.tree.range_max(self.pos[head], self.pos[a]))
            a = self.parent[head]  # type: ignore[assignment]
            crossed += 1
        lo, hi = sorted((self.pos[a], self.pos[b]))
        best = max(best, self.tree.range_max(lo, hi))
        self.max_chains_crossed = max(self.max_chains_crossed, crossed + 1)
        return best

    def update(self, node: str, value: float) -> None:
        if node not in self.pos:
            raise Missing(f"node '{node}' is not in the tree")
        self.tree.update(self.pos[node], value)

    def chain_count(self) -> int:
        return len(set(self.head.values()))

    def note(self) -> str:
        return (
            f"{self.chain_count()} chain(s) over {self.graph.node_count()} node(s), most "
            f"chains crossed by a query {self.max_chains_crossed}; a path is one chain, a "
            "broom is many, and the crossing count is the logarithm promised"
        )
