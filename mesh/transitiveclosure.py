"""Transitive closure and reduction: all the reachability, or only the necessary edges.

The transitive closure of a directed graph adds an edge from every node to
every node it can reach by any path, so that reachability becomes a single
edge lookup. It is the precomputed answer to can-A-reach-B for every pair,
the thing a permissions system wants when roles inherit from roles, and a
dependency tracker wants when it asks what a change could affect. The
closure is computed by a reachability search from each node, breadth-first
over the out-edges, marking everything reached, in nodes times edges
total; the Floyd-Warshall pattern with boolean or in place of min does the
same in nodes cubed and is the classic statement of it. The transitive
reduction is the opposite operation and the more interesting one: the
smallest graph with the same closure, keeping only edges that are not
implied by other paths. An edge from u to v is redundant if v is reachable
from u by some other route, through a node other than u or v, and dropping
it changes nothing about who reaches whom. For a directed acyclic graph the
reduction is unique and is what a dependency diagram should show, the
direct dependencies only, without the clutter of edges that merely repeat
what a chain already says. For a graph with cycles the reduction is not
unique, since any node in a cycle could be dropped from a path in several
ways, so this engine computes the reduction only for a DAG and refuses
otherwise. The computation checks each edge u to v by asking whether v is
reachable from u after removing that edge, which is the closure restricted
to paths of length at least two. The engine returns the closure as a
reachability set per node, the reduction as an edge list, and reports how
many edges the reduction removed, because a large fraction removed is a
graph drawn with every implied edge spelled out, and the reduction is the
drawing it should have been.
"""

from __future__ import annotations

from collections import deque

from mesh.errors import Invalid
from mesh.graph import Graph
from mesh.toposort import TopologicalSort


class TransitiveClosure:
    def __init__(self, graph: Graph) -> None:
        if not graph.directed:
            raise Invalid("transitive closure is a directed-graph notion")
        self.graph = graph
        self.reach: dict[str, set[str]] = {n: self._reachable(n) for n in graph.nodes()}

    def _reachable(self, source: str, skip_edge: tuple[str, str] | None = None) -> set[str]:
        # BFS over out-edges, optionally pretending one edge is absent
        seen: set[str] = set()
        queue: deque[str] = deque([source])
        while queue:
            node = queue.popleft()
            for nbr in self.graph.neighbors(node):
                if skip_edge == (node, nbr):
                    continue
                if nbr not in seen:
                    seen.add(nbr)
                    queue.append(nbr)
        return seen

    def reaches(self, u: str, v: str) -> bool:
        return v in self.reach[u]

    def reduction(self) -> list[tuple[str, str]]:
        TopologicalSort(self.graph).order()  # refuses a cyclic graph
        kept: list[tuple[str, str]] = []
        for u, v, _w in self.graph.edges():
            # the edge is redundant if v is still reachable without it
            if v not in self._reachable(u, skip_edge=(u, v)):
                kept.append((u, v))
        return kept

    def removed_count(self) -> int:
        return self.graph.edge_count() - len(self.reduction())

    def note(self) -> str:
        removed = self.removed_count()
        total = self.graph.edge_count() or 1
        return (
            f"reduction removed {removed} of {self.graph.edge_count()} edge(s) "
            f"({removed / total * 100:.0f}%); a large fraction is a graph drawn "
            "with every implied edge spelled out"
        )
