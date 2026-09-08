"""Hopcroft-Karp: maximum bipartite matching by batches of shortest augmenting paths.

A matching in a bipartite graph is a set of edges no two of which share an
endpoint, pairing left nodes with right nodes one to one, and a maximum
matching pairs as many as possible. Workers to shifts, students to
projects, any one-to-one assignment reduces to it. The engine of every
matching algorithm is the augmenting path: a path that starts at an
unmatched left node, ends at an unmatched right node, and alternates
between edges not in the matching and edges in it. Flipping such a path,
putting its non-matching edges in and taking its matching edges out, grows
the matching by exactly one, and Berge's theorem says a matching is maximum
exactly when no augmenting path exists, so finding and flipping paths until
none remain is a complete method. The naive version finds one path per
search. Hopcroft-Karp finds a whole batch per phase: a breadth-first pass
from all unmatched left nodes at once layers the graph by distance, then a
depth-first pass finds a maximal set of vertex-disjoint shortest augmenting
paths within those layers and flips them all. Because each phase strictly
increases the shortest augmenting path length and that length cannot exceed
the node count, the number of phases is bounded by the square root of the
node count, which is what lifts the running time from nodes times edges to
root-nodes times edges. The matcher takes the two sides and the edges
between them, runs phases until no augmenting path remains, returns the
matching and its size, and refuses an edge that does not cross the sides.
It reports the matching size against the smaller side, because a matching
that saturates the smaller side is perfect for that side and a shortfall
means some node there has no available partner, a deficiency the pairing
cannot fix.
"""

from __future__ import annotations

import math
from collections import deque

from mesh.errors import Invalid


class HopcroftKarp:
    def __init__(self, left: list[str], right: list[str]) -> None:
        if set(left) & set(right):
            raise Invalid("a node cannot be on both sides of a bipartite graph")
        self.left = list(left)
        self.right = list(right)
        self._right_set = set(right)
        self.adj: dict[str, list[str]] = {u: [] for u in self.left}
        self.match_left: dict[str, str | None] = dict.fromkeys(self.left)
        self.match_right: dict[str, str | None] = dict.fromkeys(self.right)
        self.phases = 0

    def add_edge(self, u: str, v: str) -> None:
        if u not in self.adj or v not in self._right_set:
            raise Invalid(f"edge {u}-{v} does not cross from the left to the right side")
        self.adj[u].append(v)

    def _bfs(self, dist: dict[str, float]) -> bool:
        # layer the graph from every free left node at once
        queue: deque[str] = deque()
        for u in self.left:
            if self.match_left[u] is None:
                dist[u] = 0
                queue.append(u)
            else:
                dist[u] = math.inf
        found_free = False
        while queue:
            u = queue.popleft()
            for v in self.adj[u]:
                w = self.match_right[v]
                if w is None:
                    found_free = True
                elif dist[w] == math.inf:
                    dist[w] = dist[u] + 1
                    queue.append(w)
        return found_free

    def _dfs(self, u: str, dist: dict[str, float]) -> bool:
        # follow only edges that step to the next layer
        for v in self.adj[u]:
            w = self.match_right[v]
            if w is None or (dist[w] == dist[u] + 1 and self._dfs(w, dist)):
                self.match_left[u] = v
                self.match_right[v] = u
                return True
        dist[u] = math.inf  # dead end this phase, prune it
        return False

    def solve(self) -> int:
        size = 0
        dist: dict[str, float] = {}
        while self._bfs(dist):
            self.phases += 1
            for u in self.left:
                if self.match_left[u] is None and self._dfs(u, dist):
                    size += 1
        return size

    def matching(self) -> dict[str, str]:
        return {u: v for u, v in self.match_left.items() if v is not None}

    def note(self) -> str:
        size = len(self.matching())
        smaller = min(len(self.left), len(self.right))
        return (
            f"matched {size} of the smaller side's {smaller} in {self.phases} "
            "phase(s); a shortfall is a node with no available partner"
        )
