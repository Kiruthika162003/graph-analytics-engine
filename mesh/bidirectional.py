"""Bidirectional BFS: search from both ends and meet in the middle.

A breadth-first search from a source to a target explores rings of growing
radius until the ring reaches the target, and the number of nodes it
touches grows with the branching factor raised to the distance. A search
from both ends at once, one ring from the source and one from the target,
alternating, needs each side to reach only half the distance before the
two frontiers touch, and because the cost is exponential in the radius,
two searches of half the radius together touch far fewer nodes than one
search of the full radius: the branching factor to the half power, twice,
against the branching factor to the full power. On a graph with any real
branching, that is the difference between a search that is fast and one
that visits most of the graph. The correctness subtlety is the meeting
rule. The two frontiers touching at a node does not by itself give the
shortest path, because the first touch found might be at a node that is
one ring farther on one side than a better meeting point on the other; so
the search expands whole rings at a time, always the smaller frontier
first to keep the work balanced, and when a node is discovered by both
sides it checks every node discovered by both and takes the one with the
smallest combined distance. For an undirected graph the backward search
follows the same edges; for a directed graph it must follow edges
backward, into the target, which the engine does by consulting the reverse
graph. The search returns the shortest distance and path, refuses a
missing endpoint, reports unreachability, and reports the nodes it
expanded against what a plain BFS would have expanded from the source,
because that ratio is the saving the meet-in-the-middle bought, and on a
long thin path it is near one, where on a branching graph it is large.
"""

from __future__ import annotations

from collections import deque

from mesh.bfs import BFS
from mesh.errors import Missing, Unreachable
from mesh.graph import Graph


class BidirectionalBFS:
    def __init__(self, graph: Graph, source: str, target: str) -> None:
        for endpoint in (source, target):
            if not graph.has_node(endpoint):
                raise Missing(f"'{endpoint}' is not in the graph")
        self.graph = graph
        self.source = source
        self.target = target
        self._backward_graph = graph.reverse() if graph.directed else graph
        self.expanded = 0
        self._dist_f: dict[str, int] = {source: 0}
        self._dist_b: dict[str, int] = {target: 0}
        self._par_f: dict[str, str | None] = {source: None}
        self._par_b: dict[str, str | None] = {target: None}
        self.meeting: str | None = self._run()

    def _run(self) -> str | None:
        if self.source == self.target:
            return self.source
        front: deque[str] = deque([self.source])
        back: deque[str] = deque([self.target])
        while front and back:
            # expand the smaller frontier, a whole ring at a time
            if len(front) <= len(back):
                met = self._expand_ring(
                    front, self.graph, self._dist_f, self._par_f, self._dist_b
                )
            else:
                met = self._expand_ring(
                    back, self._backward_graph, self._dist_b, self._par_b, self._dist_f
                )
            if met:
                # among all nodes seen by both sides, take the best combined distance
                both = set(self._dist_f) & set(self._dist_b)
                return min(both, key=lambda n: (self._dist_f[n] + self._dist_b[n], n))
        return None

    def _expand_ring(
        self,
        frontier: deque[str],
        graph: Graph,
        dist: dict[str, int],
        parent: dict[str, str | None],
        other: dict[str, int],
    ) -> bool:
        met = False
        for _ in range(len(frontier)):
            node = frontier.popleft()
            self.expanded += 1
            for nbr in graph.neighbors(node):
                if nbr not in dist:
                    dist[nbr] = dist[node] + 1
                    parent[nbr] = node
                    frontier.append(nbr)
                    if nbr in other:
                        met = True
        return met

    def distance(self) -> int:
        if self.meeting is None:
            raise Unreachable(f"'{self.target}' is unreachable from '{self.source}'")
        return self._dist_f[self.meeting] + self._dist_b[self.meeting]

    def path(self) -> list[str]:
        if self.meeting is None:
            raise Unreachable(f"'{self.target}' is unreachable from '{self.source}'")
        left: list[str] = []
        cur: str | None = self.meeting
        while cur is not None:
            left.append(cur)
            cur = self._par_f[cur]
        left.reverse()
        cur = self._par_b[self.meeting]
        while cur is not None:
            left.append(cur)
            cur = self._par_b[cur]
        return left

    def plain_expanded(self) -> int:
        # what a one-sided BFS would touch to reach the target's ring
        one_sided = BFS(self.graph, self.source)
        if self.target not in one_sided.distance:
            return len(one_sided.distance)
        radius = one_sided.distance[self.target]
        return sum(1 for d in one_sided.distance.values() if d < radius) + 1

    def note(self) -> str:
        plain = self.plain_expanded()
        return (
            f"expanded {self.expanded} node(s) against {plain} for one-sided BFS; "
            "the saving is large on a branching graph and near nothing on a thin path"
        )
