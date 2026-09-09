"""Pathwidth and vertex separation: laying a graph out in a line with the fewest live nodes.

Order the nodes in a line and, at each cut between positions, count
the nodes on the left that still have a neighbor on the right; the
largest such count is the vertex separation of the ordering, and the
smallest over all orderings is the vertex separation number of the
graph, which Kinnersley showed equals its pathwidth. It is the width
a register allocator or a sweep-line algorithm needs when it must
process nodes one at a time and keep every node with unfinished
business live. A path has pathwidth one, a cycle two, a complete
graph n minus 1, and a star one, since the hub is the only node ever
live. Pathwidth is at least treewidth and the two can differ by a
logarithmic factor on trees, which is why a complete binary tree of
depth d has pathwidth about d over 2 while its treewidth is one. The
engine computes the exact number on small graphs by dynamic
programming over subsets in the manner of Bodlaender and Kloks: the
cost of a prefix set S is the number of nodes of S with a neighbor
outside S, and the best ordering of S is the best over the last node
added, taking the larger of the prefix cost and the cost of S itself.
It also gives a cheap upper bound from the same breadth-first
level ordering the bandwidth module uses, and checks the bound never
falls below the exact answer. Above sixteen nodes the exact method is
refused; a directed graph is refused.
"""

from __future__ import annotations

from collections import deque
from math import inf

from mesh.errors import Invalid
from mesh.graph import Graph


class Pathwidth:
    def __init__(self, graph: Graph) -> None:
        if graph.directed:
            raise Invalid("pathwidth is read on an undirected graph")
        self.graph = graph
        self.nodes = graph.nodes()
        self.index = {n: i for i, n in enumerate(self.nodes)}
        self.masks = [0] * len(self.nodes)
        for u, v, _w in graph.edges():
            self.masks[self.index[u]] |= 1 << self.index[v]
            self.masks[self.index[v]] |= 1 << self.index[u]

    def boundary(self, subset: int) -> int:
        # nodes inside the subset with at least one neighbor outside it
        return sum(
            1 for i, m in enumerate(self.masks) if subset >> i & 1 and m & ~subset
        )

    def separation(self, order: list[str]) -> int:
        subset = 0
        worst = 0
        for node in order[:-1]:
            subset |= 1 << self.index[node]
            worst = max(worst, self.boundary(subset))
        return worst

    def exact(self) -> int:
        n = len(self.nodes)
        if n > 16:
            raise Invalid("exact pathwidth runs over every subset; keep it to sixteen nodes")
        if n == 0:
            return 0
        full = (1 << n) - 1
        best = [inf] * (full + 1)
        best[0] = 0
        cost = [0] * (full + 1)
        for subset in range(1, full + 1):
            cost[subset] = self.boundary(subset)
        for subset in range(1, full + 1):
            here = cost[subset]
            value = inf
            rest = subset
            while rest:
                bit = rest & -rest
                rest ^= bit
                value = min(value, max(best[subset ^ bit], here))
            best[subset] = value
        # the full set has an empty boundary; the answer is the best over its last node
        return int(best[full])

    def level_order_bound(self) -> int:
        if not self.nodes:
            return 0
        best = inf
        degree = {n: self.graph.degree(n) for n in self.nodes}
        starts = sorted(self.nodes, key=lambda n: (degree[n], n))[:4]
        for start in starts:
            order = [start]
            seen = {start}
            queue = deque([start])
            while queue:
                node = queue.popleft()
                nbrs = sorted(self.graph.neighbors(node), key=lambda x: (degree[x], x))
                for m in nbrs:
                    if m not in seen:
                        seen.add(m)
                        order.append(m)
                        queue.append(m)
            order.extend(n for n in self.nodes if n not in seen)
            best = min(best, self.separation(order))
        return int(best)

    def note(self) -> str:
        exact = self.exact()
        bound = self.level_order_bound()
        gap = "meets the exact value" if bound == exact else f"is {bound - exact} above it"
        return f"pathwidth {exact}; the level-order bound of {bound} {gap}"
