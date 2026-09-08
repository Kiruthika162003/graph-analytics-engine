"""Girth: the length of the shortest cycle, or the verdict that there is none.

The girth of a graph is the number of edges in its shortest cycle. A tree
has no cycle and its girth is taken as infinite; a triangle-free graph has
girth at least four; a graph with girth much larger than its average
degree would suggest is locally tree-like, which is the property that
makes message-passing and belief-propagation methods behave, and the
property that error-correcting code designers chase. Finding the girth is
a breadth-first search from every node with one extra observation. When a
BFS from a root meets an edge whose two endpoints are both already
discovered and the edge is not the tree edge that discovered one of them,
a cycle exists through the root's search tree, and its length is at most
the sum of the two endpoints' depths plus one. The shortest cycle through
any node will be found by the BFS rooted at a node on that cycle, as the
first such non-tree edge it encounters by depth, so the minimum over all
roots of the first non-tree-edge length is the girth. Each BFS can stop
as soon as its frontier depth exceeds half the best cycle found so far,
since no shorter cycle can still appear, which prunes the later searches
on graphs with a short cycle. For a directed graph the same idea applies
with directed edges and the non-tree edge closing back to an ancestor,
but this engine restricts itself to the undirected case, where the
argument is clean and the answer is exact. The finder returns the girth,
whether the graph is acyclic, one shortest cycle as a node list, and
reports the girth against the count of edges, because a graph with many
edges and a large girth is sparse in cycles in a way density alone does
not show, the locally tree-like case.
"""

from __future__ import annotations

import math
from collections import deque

from mesh.errors import Invalid
from mesh.graph import Graph


class Girth:
    def __init__(self, graph: Graph) -> None:
        if graph.directed:
            raise Invalid("this girth finder handles undirected graphs")
        self.graph = graph
        self.girth = math.inf
        self.cycle: list[str] = []
        self._run()

    def _run(self) -> None:
        for root in self.graph.nodes():
            self._search(root)

    def _search(self, root: str) -> None:
        depth = {root: 0}
        parent: dict[str, str | None] = {root: None}
        queue: deque[str] = deque([root])
        while queue:
            u = queue.popleft()
            if 2 * depth[u] + 1 >= self.girth:
                return  # no cycle through here can beat the best found
            for v in self.graph.neighbors(u):
                if v not in depth:
                    depth[v] = depth[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif parent[u] != v:
                    # a non-tree edge closes a cycle through the search tree
                    length = depth[u] + depth[v] + 1
                    if length < self.girth:
                        self.girth = length
                        self.cycle = self._reconstruct(u, v, parent)

    @staticmethod
    def _reconstruct(u: str, v: str, parent: dict[str, str | None]) -> list[str]:
        up_u: list[str] = []
        cur: str | None = u
        while cur is not None:
            up_u.append(cur)
            cur = parent[cur]
        up_v: list[str] = []
        cur = v
        while cur is not None and cur not in up_u:
            up_v.append(cur)
            cur = parent[cur]
        meet = cur if cur is not None else up_u[-1]
        return up_u[: up_u.index(meet) + 1] + list(reversed(up_v))

    def is_acyclic(self) -> bool:
        return self.girth == math.inf

    def note(self) -> str:
        if self.is_acyclic():
            return f"acyclic: {self.graph.edge_count()} edge(s) and no cycle at all"
        return (
            f"girth {self.girth} over {self.graph.edge_count()} edge(s); many edges "
            "with a large girth is a locally tree-like graph"
        )
