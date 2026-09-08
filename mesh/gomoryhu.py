"""Gomory-Hu tree: every pair's minimum cut from n minus one flow computations.

There are n choose 2 pairs of nodes and asking a max-flow for each is
wasteful, because an undirected graph has at most n minus one distinct
minimum cut values. Gomory and Hu showed that a single weighted tree on
the same nodes carries all of them: the minimum cut between any two
nodes equals the lightest edge on the tree path between them, and the
two sides of that tree edge, when removed, are a minimum cut in the
original graph. The Gusfield form of the construction is the simple
one. Start with every node's tree parent set to the first node. For
each other node in order, compute a minimum cut between it and its
current parent; every node with a larger index whose parent is the same
and which lands on the node's side of the cut is reparented to the
node; the node's tree edge to its parent takes the cut's value. Each
step is one flow computation, and the edge values are min cuts by
construction. The engine builds the tree that way on top of the min cut
module, answers pairwise queries by walking the tree path for its
lightest edge, and checks a sample of answers against a direct flow
computation, which is the test that matters, because the tree is only
worth having if it never disagrees with the flow it summarises. A
directed graph is refused, and a query naming an absent node is refused
with the name.
"""

from __future__ import annotations

from mesh.errors import Invalid, Missing
from mesh.graph import Graph
from mesh.mincut import MinCut


class GomoryHu:
    def __init__(self, graph: Graph) -> None:
        if graph.directed:
            raise Invalid("a Gomory-Hu tree summarises an undirected graph")
        self.graph = graph
        self.nodes = graph.nodes()
        self._known = set(self.nodes)
        self.flow_graph = self._directed()
        self.parent: dict[str, str] = {}
        self.weight: dict[str, float] = {}
        self.flows = 0
        self._build()

    def _directed(self) -> Graph:
        # the flow modules want arcs, so every edge becomes a pair with the same capacity
        d = Graph(directed=True)
        for node in self.nodes:
            d.add_node(node)
        for u, v, w in self.graph.edges():
            d.add_edge(u, v, w)
            d.add_edge(v, u, w)
        return d

    def _build(self) -> None:
        if not self.nodes:
            return
        root = self.nodes[0]
        for node in self.nodes[1:]:
            self.parent[node] = root
        for node in self.nodes[1:]:
            target = self.parent[node]
            cut = MinCut(self.flow_graph, node, target)
            self.flows += 1
            self.weight[node] = cut.value
            # every other node hanging off the same parent on this node's side moves
            # under this node; the first version looked only at later nodes, which
            # gives an equivalent flow tree whose edges are not all real cuts
            for other in self.nodes:
                if other == node:
                    continue
                if self.parent.get(other) == target and other in cut.source_side:
                    self.parent[other] = node
            # Gusfield's swap: when the parent's own parent sits on this node's side,
            # this node takes the parent's place in the tree, which turns the
            # equivalent flow tree into a cut tree whose edges are real cuts
            grand = self.parent.get(target)
            if grand is not None and grand in cut.source_side:
                self.parent[node] = grand
                self.parent[target] = node
                self.weight[node] = self.weight[target]
                self.weight[target] = cut.value

    def side(self, node: str) -> set[str]:
        # the nodes that fall with this node when its edge to its parent is removed
        children: dict[str, list[str]] = {n: [] for n in self.nodes}
        for child, parent in self.parent.items():
            children[parent].append(child)
        seen = {node}
        stack = [node]
        while stack:
            for child in children[stack.pop()]:
                if child not in seen:
                    seen.add(child)
                    stack.append(child)
        return seen

    def edge_is_a_real_cut(self, node: str) -> bool:
        inside = self.side(node)
        crossing = sum(
            w for u, v, w in self.graph.edges() if (u in inside) != (v in inside)
        )
        return crossing == self.weight[node]

    def tree(self) -> Graph:
        t = Graph()
        for node in self.nodes:
            t.add_node(node)
        for node, parent in self.parent.items():
            t.add_edge(node, parent, self.weight[node])
        return t

    def _path_to_root(self, node: str) -> list[str]:
        chain = [node]
        while chain[-1] in self.parent:
            chain.append(self.parent[chain[-1]])
        return chain

    def min_cut(self, a: str, b: str) -> float:
        for name in (a, b):
            if name not in self._known:
                raise Missing(f"'{name}' is not a node of the graph")
        if a == b:
            return 0.0
        up_a = self._path_to_root(a)
        up_b = self._path_to_root(b)
        common = next(n for n in up_a if n in set(up_b))
        lightest = float("inf")
        for chain in (up_a, up_b):
            for node in chain:
                if node == common:
                    break
                lightest = min(lightest, self.weight[node])
        return lightest

    def agrees_with_flow(self, a: str, b: str) -> bool:
        return self.min_cut(a, b) == MinCut(self.flow_graph, a, b).value

    def note(self) -> str:
        return (
            f"Gomory-Hu tree over {len(self.nodes)} node(s) from {self.flows} flow(s); "
            f"distinct cut values {sorted(set(self.weight.values()))}"
        )
