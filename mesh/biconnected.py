"""Biconnected components: the blocks that survive any single node failure.

Two-edge-connectivity asks what survives a cut cable; biconnectivity asks
the harder question of what survives a dead router. A graph is
biconnected if removing any single node leaves it connected, and every
graph decomposes into maximal biconnected pieces, the blocks, which are
glued together at articulation points. A block is a set of edges rather
than a set of nodes, because an articulation point belongs to every block
it joins: the node itself sits in several blocks, but each edge sits in
exactly one. Inside a block any two nodes lie on a common simple cycle,
which is what makes it robust, and the blocks with their articulation
points form the block-cut tree, alternating block nodes and cut nodes,
whose structure says exactly how the graph falls apart under node
failure. The decomposition runs alongside the articulation search. Edges
are pushed onto a stack as the DFS traverses them, and when the search
finishes a child whose low value reaches at least the parent's discovery
time, the parent is an articulation point and every edge on the stack
down to the tree edge into that child is popped as one block. The edges
left on the stack at the end of a root's search form the last block. An
isolated edge is its own block, a bridge, and a triangle is one block of
three edges. The decomposer returns the blocks as edge sets, the blocks
containing a node, the articulation points, and the block-cut tree, and
it refuses a directed graph. It reports the block count against the
articulation count, because a graph that is a single block has no node
whose failure splits it, while many small blocks strung along many cut
nodes is a chain of fragile joints, the shape a network designer wants
to see before a failure draws it for them.
"""

from __future__ import annotations

from mesh.errors import Invalid, Missing
from mesh.graph import Graph


class Biconnected:
    def __init__(self, graph: Graph) -> None:
        if graph.directed:
            raise Invalid("biconnected components are defined on undirected graphs")
        self.graph = graph
        self._disc: dict[str, int] = {}
        self._low: dict[str, int] = {}
        self._time = 0
        self._edge_stack: list[tuple[str, str]] = []
        self.blocks: list[set[frozenset[str]]] = []
        self.cut_nodes: set[str] = set()
        for start in graph.nodes():
            if start not in self._disc:
                self._explore(start)

    def _pop_block(self, until: tuple[str, str]) -> None:
        block: set[frozenset[str]] = set()
        while self._edge_stack:
            edge = self._edge_stack.pop()
            block.add(frozenset(edge))
            if edge == until:
                break
        if block:
            self.blocks.append(block)

    def _explore(self, root: str) -> None:
        self._disc[root] = self._low[root] = self._time
        self._time += 1
        root_children = 0
        stack: list[tuple[str, str | None, list[str]]] = [
            (root, None, list(self.graph.neighbors(root)))
        ]
        while stack:
            node, parent, pending = stack[-1]
            if pending:
                nbr = pending.pop()
                if nbr == parent:
                    continue
                if nbr in self._disc:
                    if self._disc[nbr] < self._disc[node]:
                        self._edge_stack.append((node, nbr))  # a back edge
                        self._low[node] = min(self._low[node], self._disc[nbr])
                    continue
                self._disc[nbr] = self._low[nbr] = self._time
                self._time += 1
                self._edge_stack.append((node, nbr))
                if node == root:
                    root_children += 1
                stack.append((nbr, node, list(self.graph.neighbors(nbr))))
            else:
                stack.pop()
                if parent is None:
                    continue
                self._low[parent] = min(self._low[parent], self._low[node])
                if self._low[node] >= self._disc[parent]:
                    # the parent separates this subtree: pop its block
                    if parent != root or root_children >= 2:
                        self.cut_nodes.add(parent)
                    self._pop_block((parent, node))
        if root_children >= 2:
            self.cut_nodes.add(root)
        elif root_children == 1:
            self.cut_nodes.discard(root)

    def blocks_of(self, node: str) -> list[set[frozenset[str]]]:
        if not self.graph.has_node(node):
            raise Missing(f"node '{node}' is not in the graph")
        return [b for b in self.blocks if any(node in e for e in b)]

    def is_biconnected(self) -> bool:
        return len(self.blocks) == 1 and not self.cut_nodes

    def block_cut_tree(self) -> Graph:
        tree = Graph(directed=False)
        for i in range(len(self.blocks)):
            tree.add_node(f"B{i}")
        for c in self.cut_nodes:
            tree.add_node(f"C:{c}")
        for i, block in enumerate(self.blocks):
            for c in self.cut_nodes:
                if any(c in e for e in block):
                    tree.add_edge(f"B{i}", f"C:{c}")
        return tree

    def note(self) -> str:
        return (
            f"{len(self.blocks)} block(s) glued at {len(self.cut_nodes)} cut node(s); "
            "one block means no node's failure splits the graph, many small blocks "
            "along many cut nodes is a chain of fragile joints"
        )
