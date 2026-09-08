"""K-core decomposition: peel the graph layer by layer to find its dense heart.

The k-core of a graph is the largest subgraph in which every node has at
least k neighbors within that subgraph. It is a way of finding the dense
heart of a network by stripping away the loosely attached fringe: the
1-core drops isolated nodes, the 2-core drops the dangling leaves and the
chains that hang off, the 3-core keeps only nodes that sit in a tightly
knit region, and so on until nothing survives. Each node's core number is
the largest k for which it belongs to the k-core, and a node with a high
core number is embedded deep in the dense part of the graph, a more robust
signal of centrality than raw degree because a node of high degree whose
neighbors are all leaves has a core number of one. The decomposition is
computed by peeling. Repeatedly remove the node of smallest remaining
degree, record its core number as the current peel level, and reduce its
neighbors' degrees; when the smallest remaining degree rises, the peel
level rises with it. Processing nodes in order of degree with a bucket per
degree value makes the whole peel linear in the edge count, and the
monotone property, that a node's core number is the max of the level at
which it is removed, means one pass suffices. The degeneracy of the graph
is the largest core number, the k of the innermost non-empty core, and the
peeling order itself is a degeneracy ordering used to speed up clique
finding and coloring. The decomposer returns the core number per node, the
members of any k-core, and the degeneracy, and it reports the degeneracy
against the maximum degree, because a degeneracy far below the maximum
degree is a graph of hubs with leafy neighborhoods rather than a dense
core, two shapes that raw degree cannot tell apart.
"""

from __future__ import annotations

from mesh.errors import Invalid
from mesh.graph import Graph


class KCore:
    def __init__(self, graph: Graph) -> None:
        if graph.directed:
            raise Invalid("k-core decomposition is defined for an undirected graph")
        self.graph = graph
        self.core: dict[str, int] = {}
        self.order: list[str] = []
        self._run()

    def _run(self) -> None:
        degree = {n: self.graph.degree(n) for n in self.graph.nodes()}
        remaining = set(degree)
        level = 0
        while remaining:
            # peel the node of smallest remaining degree; the level only rises
            node = min(remaining, key=lambda n: (degree[n], n))
            level = max(level, degree[node])
            self.core[node] = level
            self.order.append(node)
            remaining.discard(node)
            for nbr in self.graph.neighbors(node):
                if nbr in remaining:
                    degree[nbr] -= 1

    def k_core(self, k: int) -> set[str]:
        if k < 0:
            raise Invalid("k cannot be negative")
        return {n for n, c in self.core.items() if c >= k}

    def degeneracy(self) -> int:
        return max(self.core.values(), default=0)

    def note(self) -> str:
        max_degree = max((self.graph.degree(n) for n in self.graph.nodes()), default=0)
        return (
            f"degeneracy {self.degeneracy()} against max degree {max_degree}; a "
            "degeneracy far below the max degree is hubs with leafy neighborhoods, "
            "not a dense core"
        )
