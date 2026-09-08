"""Second-best spanning tree: the cheapest tree that is not the cheapest one.

A network planner who has the minimum spanning tree usually wants to
know the runner-up: how much worse is the next-best design, and which
single swap gets there. The second-best minimum spanning tree differs
from the best by exactly one edge exchange, a fact that makes it cheap
to find where enumerating trees would be hopeless. Take the minimum
tree. For every edge not in it, adding that edge closes one cycle, the
tree path between its endpoints plus the edge itself; removing the
heaviest tree edge on that path gives a new spanning tree whose weight
is the old weight plus the new edge minus that heaviest path edge. The
second-best tree is the cheapest of these exchanges, and the proof that
no tree differing in two or more edges can beat it is the same exchange
argument that proves the minimum tree optimal, applied once more. The
cost is the path-maximum query per non-tree edge, and the engine uses
its heavy-light decomposition for that, so each query is a logarithm
squared rather than a walk. When the exchange leaves the weight
unchanged the minimum tree was not unique, and the engine says so rather
than reporting a runner-up that ties the winner. A tree with no
non-tree edges, the input already being a tree, has no second-best
tree at all and the engine refuses honestly. The finder returns the
second-best weight, the edge added and the edge dropped, whether the
minimum was unique, and reports the gap between best and second-best,
because that gap is how much margin the chosen design has over the
alternative and a gap of zero is a choice that was a coin flip.
"""

from __future__ import annotations

import math

from mesh.errors import Invalid
from mesh.graph import Graph
from mesh.hld import HeavyLight
from mesh.kruskal import Kruskal


class SecondBestMST:
    def __init__(self, graph: Graph) -> None:
        if graph.directed:
            raise Invalid("a spanning tree is defined for an undirected graph")
        base = Kruskal(graph)
        if not base.spans():
            raise Invalid("the graph is disconnected; there is no spanning tree to improve on")
        self.graph = graph
        self.best = base.total_weight()
        self.tree_edges = base.edges()
        tree_set = {frozenset((u, v)) for u, v, _w in self.tree_edges}
        self.non_tree = [
            (u, v, w) for u, v, w in graph.edges() if frozenset((u, v)) not in tree_set
        ]
        if not self.non_tree:
            raise Invalid("the graph is already a tree; no second-best spanning tree exists")
        self.added: tuple[str, str, float] | None = None
        self.dropped_weight = 0.0
        self.second = self._search()

    def _search(self) -> float:
        # the tree as a graph with edge weights stored on the child node
        tree = Graph()
        for n in self.graph.nodes():
            tree.add_node(n)
        for u, v, w in self.tree_edges:
            tree.add_edge(u, v, w)
        root = self.graph.nodes()[0]
        parent_weight = self._weights_toward(tree, root)
        hld = HeavyLight(tree, root, parent_weight)
        best = math.inf
        for u, v, w in self.non_tree:
            heaviest = self._path_max(hld, u, v)
            candidate = self.best + w - heaviest
            if candidate < best:
                best = candidate
                self.added = (u, v, w)
                self.dropped_weight = heaviest
        return best

    @staticmethod
    def _weights_toward(tree: Graph, root: str) -> dict[str, float]:
        # each non-root node carries the weight of the edge to its parent
        values = {root: float("-inf")}
        stack = [root]
        seen = {root}
        while stack:
            node = stack.pop()
            for nbr, w in tree.neighbors(node).items():
                if nbr not in seen:
                    seen.add(nbr)
                    values[nbr] = w
                    stack.append(nbr)
        return values

    @staticmethod
    def _path_max(hld: HeavyLight, u: str, v: str) -> float:
        # the LCA carries the weight of the edge above it, which is not on the
        # path, so exclude it by querying each side up to but not past the top
        best = float("-inf")
        a, b = u, v
        while hld.head[a] != hld.head[b]:
            if hld.depth[hld.head[a]] < hld.depth[hld.head[b]]:
                a, b = b, a
            head = hld.head[a]
            best = max(best, hld.tree.range_max(hld.pos[head], hld.pos[a]))
            a = hld.parent[head]  # type: ignore[assignment]
        if a != b:
            lo, hi = sorted((hld.pos[a], hld.pos[b]))
            best = max(best, hld.tree.range_max(lo + 1, hi))
        return best

    def unique_minimum(self) -> bool:
        return self.second > self.best

    def gap(self) -> float:
        return self.second - self.best

    def note(self) -> str:
        if not self.unique_minimum():
            return f"the minimum spanning tree at {self.best} is not unique: a tie, a coin flip"
        swap = self.added[:2] if self.added else None
        return (
            f"second-best {self.second} against best {self.best}, a margin of {self.gap()}; "
            f"swap in {swap} for the {self.dropped_weight} edge"
        )
