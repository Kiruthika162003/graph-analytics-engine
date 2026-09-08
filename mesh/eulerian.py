"""Eulerian trails: walk every edge exactly once, and the degree rule that says when.

An Eulerian circuit is a closed walk that uses every edge of a graph
exactly once and returns to its start; an Eulerian path does the same but
may end somewhere else. It is the bridges-of-Konigsberg question, and its
answer is one of the cleanest theorems in graph theory. A connected
undirected graph has an Eulerian circuit exactly when every node has even
degree, because a walk that passes through a node enters and leaves it in
pairs, consuming two edges each time, so an odd-degree node would have an
edge left over. It has an Eulerian path but not a circuit exactly when
precisely two nodes have odd degree, which must be the walk's two ends,
and it has neither when more than two are odd. The parity check decides
existence before any walking is attempted, so the engine refuses honestly
with the count of odd nodes rather than searching for a walk that cannot
exist. Finding the walk when it exists is Hierholzer's algorithm. Start
at an odd node if there are two or anywhere if there are none, and follow
unused edges greedily until stuck, which can only happen back at the start
by parity; then backtrack to any node on that walk with unused edges,
splice in a sub-walk from there, and repeat until every edge is used. Done
with an explicit stack it runs in time linear in the edge count, appending
each node to the result as its unused edges run out, so the result is the
trail in reverse. Connectivity among the nodes that carry edges is
required, since edges in a separate component can never be reached. The
finder returns the trail, states whether it is a circuit or an open path,
and reports the odd-degree node count, the number that decides everything
before a single step is taken.
"""

from __future__ import annotations

from mesh.errors import Invalid
from mesh.graph import Graph


class Eulerian:
    def __init__(self, graph: Graph) -> None:
        if graph.directed:
            raise Invalid("this Eulerian finder handles undirected graphs")
        if graph.edge_count() == 0:
            raise Invalid("a graph with no edges has no trail to walk")
        self.graph = graph
        self.odd_nodes = [n for n in graph.nodes() if graph.degree(n) % 2 == 1]
        if len(self.odd_nodes) not in (0, 2):
            raise Invalid(
                f"{len(self.odd_nodes)} node(s) have odd degree; a trail needs "
                "zero (circuit) or exactly two (path)"
            )
        self._check_edges_connected()
        self.trail = self._hierholzer()

    def _check_edges_connected(self) -> None:
        carriers = [n for n in self.graph.nodes() if self.graph.degree(n) > 0]
        seen = {carriers[0]}
        stack = [carriers[0]]
        while stack:
            node = stack.pop()
            for nbr in self.graph.neighbors(node):
                if nbr not in seen:
                    seen.add(nbr)
                    stack.append(nbr)
        if len(seen) != len(carriers):
            raise Invalid("edges lie in more than one component; no single walk covers them")

    def _hierholzer(self) -> list[str]:
        # multiset of unused edges per node, an undirected edge listed at both ends
        unused: dict[str, list[str]] = {
            n: list(self.graph.neighbors(n)) for n in self.graph.nodes()
        }
        start = self.odd_nodes[0] if self.odd_nodes else next(
            n for n in self.graph.nodes() if self.graph.degree(n) > 0
        )
        stack = [start]
        trail: list[str] = []
        while stack:
            node = stack[-1]
            if unused[node]:
                nbr = unused[node].pop()
                unused[nbr].remove(node)  # consume the edge from both ends
                stack.append(nbr)
            else:
                trail.append(stack.pop())  # no edges left here: it is finished
        trail.reverse()
        return trail

    def is_circuit(self) -> bool:
        return not self.odd_nodes

    def note(self) -> str:
        kind = "circuit" if self.is_circuit() else "open path"
        return (
            f"Eulerian {kind} over {len(self.trail) - 1} edge(s), "
            f"{len(self.odd_nodes)} odd-degree node(s); the odd count decided "
            "existence before a single step was taken"
        )
