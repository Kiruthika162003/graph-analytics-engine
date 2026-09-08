"""Cycle basis: every cycle in the graph is a combination of these few.

A connected graph with n nodes and m edges has exactly m minus n plus one
independent cycles, its cyclomatic number, and any cycle in the graph can
be written as a combination of that many basis cycles, combining by
taking the symmetric difference of edge sets. The number is what a
software engineer knows as McCabe's complexity when the graph is a
program's control flow, and what a chemist counts as rings in a
molecule. The construction is direct. Take any spanning tree of the
graph. Every edge not in the tree, a chord, closes exactly one cycle when
added to it: the chord together with the tree path between its
endpoints. Those chord cycles, one per non-tree edge, are independent,
because each contains a chord that no other contains, and they span the
cycle space, because any cycle's chords determine it. So the fundamental
cycle basis with respect to a spanning tree is the set of chord cycles,
and its size is the chord count, which is m minus n plus one for a
connected graph and m minus n plus c for c components. Different
spanning trees give different bases of the same size, and none is
canonical; the shortest total basis is a harder problem this engine
does not attempt. The builder takes a breadth-first spanning tree, uses
the tree's parent pointers to find the path between each chord's ends by
walking both up to their meeting point, returns the basis cycles as node
lists, and verifies each is a real closed walk of the graph. It refuses
a directed graph, and reports the cyclomatic number against the edge
count, because a graph whose edges are mostly chords is dense with
independent loops while one with few chords is nearly a tree.
"""

from __future__ import annotations

from mesh.errors import Invalid
from mesh.graph import Graph


class CycleBasis:
    def __init__(self, graph: Graph) -> None:
        if graph.directed:
            raise Invalid("a cycle basis here is for undirected graphs")
        self.graph = graph
        self._parent: dict[str, str | None] = {}
        self._depth: dict[str, int] = {}
        self.chords: list[tuple[str, str]] = []
        self.components = 0
        self._forest()
        self.cycles: list[list[str]] = [self._close(u, v) for u, v in self.chords]

    def _forest(self) -> None:
        # a BFS spanning forest; every edge it does not use is a chord
        tree_edges: set[frozenset[str]] = set()
        for root in self.graph.nodes():
            if root in self._parent:
                continue
            self.components += 1
            self._parent[root] = None
            self._depth[root] = 0
            queue = [root]
            while queue:
                node = queue.pop(0)
                for nbr in self.graph.neighbors(node):
                    if nbr not in self._parent:
                        self._parent[nbr] = node
                        self._depth[nbr] = self._depth[node] + 1
                        tree_edges.add(frozenset((node, nbr)))
                        queue.append(nbr)
        for u, v, _w in self.graph.edges():
            if frozenset((u, v)) not in tree_edges:
                self.chords.append((u, v))

    def _close(self, u: str, v: str) -> list[str]:
        # the chord plus the tree path between its ends, found by walking up
        up_u: list[str] = [u]
        up_v: list[str] = [v]
        a, b = u, v
        while a != b:
            if self._depth[a] >= self._depth[b]:
                a = self._parent[a]  # type: ignore[assignment]
                up_u.append(a)
            else:
                b = self._parent[b]  # type: ignore[assignment]
                up_v.append(b)
        return up_u + list(reversed(up_v[:-1]))

    def cyclomatic_number(self) -> int:
        return self.graph.edge_count() - self.graph.node_count() + self.components

    def is_closed_walk(self, cycle: list[str]) -> bool:
        if len(cycle) < 3:
            return False
        return all(
            self.graph.has_edge(cycle[i], cycle[(i + 1) % len(cycle)])
            for i in range(len(cycle))
        )

    def note(self) -> str:
        m = self.graph.edge_count() or 1
        return (
            f"{self.cyclomatic_number()} independent cycle(s) from "
            f"{len(self.chords)} chord(s) among {self.graph.edge_count()} edge(s) "
            f"({len(self.chords) / m * 100:.0f}% chords); mostly chords is dense "
            "with loops, few chords is nearly a tree"
        )
