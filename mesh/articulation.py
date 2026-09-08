"""Articulation points and bridges: the single failures that split a graph.

An articulation point is a node whose removal disconnects the graph, and a
bridge is an edge whose removal does the same. They are the single points
of failure of a network: the one router every path between two regions
crosses, the one cable whose cut isolates a building. Finding them naively
means removing each node or edge in turn and re-checking connectivity,
nodes times edges of work. One depth-first search finds them all, by the
same low-link idea that Tarjan uses for strongly connected components.
Each node is stamped with a discovery time, and its low value is the
earliest discovery time reachable from its DFS subtree using at most one
back edge. A node v with a child c in the DFS tree is an articulation point
if the low value of c is at least the discovery time of v, because then
nothing in c's subtree has a back edge climbing above v, so removing v cuts
that subtree off. The tree edge v to c is a bridge under the strictly
stronger condition that c's low value exceeds v's discovery time, because
if it merely equalled it there would be a back edge from the subtree to v
itself, offering a second route across. The root of the DFS tree is the
one exception to the articulation rule: it has no parent to be cut from,
so it is an articulation point only when it has two or more DFS children,
each of which would become its own piece. In an undirected graph the edge
back to the parent must be skipped when computing the low value, or every
tree edge would falsely look like it has a back edge. The finder returns
the articulation points and bridges, and reports their counts, because a
graph with none of either is two-connected and survives any single
failure, while one with many has a fragility map a designer wants drawn.
"""

from __future__ import annotations

from mesh.errors import Invalid
from mesh.graph import Graph


class Articulation:
    def __init__(self, graph: Graph) -> None:
        if graph.directed:
            raise Invalid("articulation points and bridges are defined on undirected graphs")
        self.graph = graph
        self._disc: dict[str, int] = {}
        self._low: dict[str, int] = {}
        self._time = 0
        self.points: set[str] = set()
        self.bridges: set[frozenset[str]] = set()
        self._run()

    def _run(self) -> None:
        for start in self.graph.nodes():
            if start not in self._disc:
                self._explore(start)

    def _explore(self, root: str) -> None:
        # iterative DFS: each frame is (node, parent, iterator over neighbors)
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
                    continue  # the edge back to the parent is not a back edge
                if nbr in self._disc:
                    self._low[node] = min(self._low[node], self._disc[nbr])
                else:
                    self._disc[nbr] = self._low[nbr] = self._time
                    self._time += 1
                    if node == root:
                        root_children += 1
                    stack.append((nbr, node, list(self.graph.neighbors(nbr))))
            else:
                stack.pop()
                if parent is None:
                    continue
                self._low[parent] = min(self._low[parent], self._low[node])
                if self._low[node] >= self._disc[parent] and parent != root:
                    self.points.add(parent)
                if self._low[node] > self._disc[parent]:
                    self.bridges.add(frozenset((parent, node)))
        if root_children >= 2:
            self.points.add(root)

    def is_two_connected(self) -> bool:
        return not self.points and not self.bridges

    def note(self) -> str:
        return (
            f"{len(self.points)} articulation point(s), {len(self.bridges)} "
            "bridge(s); none of either means the graph survives any single "
            "failure, many is a fragility map"
        )
