"""Reverse delete: reach a minimum spanning tree by discarding the heaviest edges you can.

Kruskal adds edges lightest first and skips any that would close a
cycle. Reverse delete runs the same idea backward: start with every
edge, consider them heaviest first, and remove each one unless removing
it would disconnect the graph. What survives is a minimum spanning
tree, by the cycle property this time rather than the cut property: the
heaviest edge on any cycle belongs to no minimum spanning tree, and an
edge that can be removed without disconnecting is exactly one lying on
a cycle, where it is the heaviest edge of that cycle not yet removed.
The two algorithms are mirror images and on a connected graph they must
produce trees of the same total weight, which the engine checks. The
connectivity test after each tentative removal is the cost, a
breadth-first search per edge, so reverse delete is edges times nodes
plus edges, slower than Kruskal's near-linear sort and union-find, and
it is kept here for what it demonstrates rather than for speed: that
the minimum spanning tree can be reached from above as well as from
below, and that the cycle property and the cut property are the same
fact seen from two sides. On a disconnected graph reverse delete never
removes a bridge, so it leaves a spanning forest and reports as much.
The builder returns the surviving edges and their total, counts how
many edges were deleted, and reports the deletions against the edge
count, because on a tree nothing can be deleted while on a dense graph
almost everything is, and the count is a direct reading of how far the
input was from being a tree already.
"""

from __future__ import annotations

from mesh.bfs import BFS
from mesh.errors import Invalid
from mesh.graph import Graph


class ReverseDelete:
    def __init__(self, graph: Graph) -> None:
        if graph.directed:
            raise Invalid("a spanning tree is defined for an undirected graph")
        self.graph = graph
        self.kept: list[tuple[str, str, float]] = list(graph.edges())
        self.deleted = 0
        self._prune()

    def _connected_without(self, skip: tuple[str, str, float]) -> bool:
        # rebuild without the candidate edge and check the original connectivity
        trial = Graph()
        for n in self.graph.nodes():
            trial.add_node(n)
        for u, v, w in self.kept:
            if (u, v, w) != skip:
                trial.add_edge(u, v, w)
        before = len(BFS(self.graph, skip[0]).reachable_nodes())
        after = len(BFS(trial, skip[0]).reachable_nodes())
        return after == before

    def _prune(self) -> None:
        # heaviest first; an edge goes if the graph stays as connected without it
        for edge in sorted(self.kept, key=lambda e: (-e[2], e[0], e[1])):
            if self._connected_without(edge):
                self.kept.remove(edge)
                self.deleted += 1

    def total_weight(self) -> float:
        return sum(w for _u, _v, w in self.kept)

    def spans(self) -> bool:
        return self.graph.node_count() > 0 and len(self.kept) == self.graph.node_count() - 1

    def note(self) -> str:
        return (
            f"deleted {self.deleted} of {self.graph.edge_count()} edge(s), keeping a "
            f"{'tree' if self.spans() else 'forest'} of weight {self.total_weight()}; "
            "nothing deletable means the input was already a tree"
        )
