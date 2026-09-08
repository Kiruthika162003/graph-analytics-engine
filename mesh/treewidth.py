"""Treewidth bound: how tree-like a graph is, estimated by eliminating nodes.

Many problems that are hard on general graphs become easy on trees, and
treewidth measures how far a graph is from being one: a tree has
treewidth one, a cycle two, a complete graph on n nodes n minus one,
and a grid grows with its side. Algorithms parameterized by treewidth
run in time exponential in the width but linear in the graph, so a
small width is a license to solve otherwise intractable problems
exactly. Computing the treewidth is itself NP-hard, but a good upper
bound comes from an elimination ordering. Eliminate nodes one at a time:
remove a node and connect all its remaining neighbors into a clique, so
that the paths through the removed node are still represented. The
width of an ordering is the largest neighborhood any node had at the
moment it was eliminated, and the treewidth is the minimum width over
all orderings, which is why any specific ordering gives an upper bound.
The minimum-degree heuristic picks, at each step, the node with the
fewest remaining neighbors, since eliminating a low-degree node adds a
small clique and keeps later degrees small; it is the standard cheap
heuristic and it is exact on trees, cycles, and complete graphs, which
the engine checks. The bound is honest about its direction: it can only
overestimate, so a reported width of five means the treewidth is at
most five and might be less. The estimator returns the elimination
ordering, its width, and the fill edges it added, and reports the width
against the degeneracy, which is a lower bound on treewidth, because
when the two meet the bound is exact and when they differ the true
value lies somewhere between them.
"""

from __future__ import annotations

from mesh.errors import Invalid
from mesh.graph import Graph
from mesh.kcore import KCore


class TreewidthBound:
    def __init__(self, graph: Graph) -> None:
        if graph.directed:
            raise Invalid("treewidth is defined on an undirected graph")
        if graph.node_count() == 0:
            raise Invalid("an empty graph has no treewidth to bound")
        self.graph = graph
        self.order: list[str] = []
        self.width = 0
        self.fill_edges = 0
        self._eliminate()

    def _eliminate(self) -> None:
        # working adjacency that grows with fill edges as nodes are removed
        adj = {n: set(self.graph.neighbors(n)) for n in self.graph.nodes()}
        while adj:
            node = min(adj, key=lambda n: (len(adj[n]), n))
            nbrs = adj[node]
            self.width = max(self.width, len(nbrs))
            # the remaining neighbors become a clique
            for a in nbrs:
                for b in nbrs:
                    if a < b and b not in adj[a]:
                        adj[a].add(b)
                        adj[b].add(a)
                        self.fill_edges += 1
            for a in nbrs:
                adj[a].discard(node)
            del adj[node]
            self.order.append(node)

    def lower_bound(self) -> int:
        return KCore(self.graph).degeneracy()

    def is_exact(self) -> bool:
        return self.width == self.lower_bound()

    def note(self) -> str:
        verdict = "exact" if self.is_exact() else "the truth lies between them"
        return (
            f"treewidth at most {self.width} and at least {self.lower_bound()} "
            f"({verdict}), {self.fill_edges} fill edge(s) added by elimination"
        )
