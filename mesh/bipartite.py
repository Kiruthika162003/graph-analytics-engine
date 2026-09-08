"""Bipartite check: two-color the graph, or produce the odd cycle that forbids it.

A graph is bipartite when its nodes split into two sides such that every
edge crosses between the sides and none lies within a side. Job seekers and
jobs, students and courses, any relation between two kinds of thing, is
bipartite by nature, and matching algorithms depend on knowing which nodes
are on which side. The test for it is a two-coloring: pick an uncolored
node, give it one color, give its neighbors the other, their neighbors the
first again, and so on by breadth-first search. If every edge ends up
joining two differently colored nodes, the coloring is the two-sided split
and the graph is bipartite. If the search ever finds an edge whose two
endpoints already carry the same color, the graph cannot be split, and the
reason is always the same: an odd cycle. Walking around a cycle alternates
colors, so a cycle of odd length returns to its start with the wrong color,
and a graph is bipartite exactly when it contains no odd cycle. The
checker does not just say no; it reconstructs the offending odd cycle from
the BFS parent pointers, walking up from both endpoints of the bad edge to
their common ancestor, so the caller can see the concrete cycle rather than
take the verdict on faith. Each connected component is colored separately,
because a disconnected graph is bipartite when every component is, and an
uncolored component must not inherit a color from another. The checker
reports the two sides when bipartite, refuses to hand out sides when it is
not, and returns the odd cycle found instead. It reports the side sizes,
because a lopsided split is a matching that can never cover the larger
side, a limit visible before any matching is attempted.
"""

from __future__ import annotations

from collections import deque

from mesh.errors import Invalid
from mesh.graph import Graph


class Bipartite:
    def __init__(self, graph: Graph) -> None:
        if graph.directed:
            raise Invalid("bipartiteness is checked on an undirected graph")
        self.graph = graph
        self.color: dict[str, int] = {}
        self.parent: dict[str, str | None] = {}
        self.odd_cycle: list[str] = []
        self.is_bipartite = self._run()

    def _run(self) -> bool:
        for start in self.graph.nodes():
            if start in self.color:
                continue
            # each component is colored from a fresh start
            self.color[start] = 0
            self.parent[start] = None
            queue: deque[str] = deque([start])
            while queue:
                node = queue.popleft()
                for nbr in self.graph.neighbors(node):
                    if nbr not in self.color:
                        self.color[nbr] = 1 - self.color[node]
                        self.parent[nbr] = node
                        queue.append(nbr)
                    elif self.color[nbr] == self.color[node]:
                        self.odd_cycle = self._cycle_through(node, nbr)
                        return False
        return True

    def _cycle_through(self, u: str, v: str) -> list[str]:
        # walk both endpoints up the BFS tree to their common ancestor
        up_u: list[str] = []
        up_v: list[str] = []
        seen_u: dict[str, int] = {}
        cur: str | None = u
        while cur is not None:
            seen_u[cur] = len(up_u)
            up_u.append(cur)
            cur = self.parent[cur]
        cur = v
        while cur is not None and cur not in seen_u:
            up_v.append(cur)
            cur = self.parent[cur]
        meet = cur if cur is not None else up_u[-1]
        return up_u[: seen_u[meet] + 1] + list(reversed(up_v))

    def sides(self) -> tuple[set[str], set[str]]:
        if not self.is_bipartite:
            raise Invalid(
                f"the graph is not bipartite; odd cycle {self.odd_cycle} forbids "
                "a two-sided split"
            )
        left = {n for n, c in self.color.items() if c == 0}
        right = {n for n, c in self.color.items() if c == 1}
        return left, right

    def note(self) -> str:
        if not self.is_bipartite:
            return f"not bipartite: odd cycle of length {len(self.odd_cycle)} found"
        left, right = self.sides()
        return (
            f"bipartite with sides of {len(left)} and {len(right)}; a lopsided "
            "split caps any matching at the smaller side"
        )
