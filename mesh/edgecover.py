"""Minimum edge cover: the fewest edges that touch every node, from a maximum matching.

An edge cover is a set of edges with every node at the end of at least
one of them, and a minimum edge cover is the smallest such set. It is
the shape of assigning every person to a pair with someone, where
pairs may overlap only when they must, and Gallai's identity says its
size is the node count minus the maximum matching size, because a
matching covers two nodes per edge and every node left over needs one
edge of its own. The construction is exactly that: take a maximum
matching from the blossom matcher, then for each unmatched node add
any one incident edge. Every node is then covered, the size is n minus
the matching's size, and no smaller cover exists because a smaller
cover would contain a larger matching. A node with no edges at all can
never be covered, so a graph with an isolated node has no edge cover
and the module refuses it by name rather than returning a set that
misses someone. The engine verifies the cover touches every node,
checks Gallai's identity, and on small graphs can be checked against
enumerating every edge subset, which is the test that establishes the
identity was not just assumed. A directed graph is refused.
"""

from __future__ import annotations

from mesh.blossom import Blossom
from mesh.errors import Invalid
from mesh.graph import Graph


class EdgeCover:
    def __init__(self, graph: Graph) -> None:
        if graph.directed:
            raise Invalid("an edge cover is defined on an undirected graph")
        for node in graph.nodes():
            if graph.degree(node) == 0:
                raise Invalid(f"'{node}' has no edge, so no edge cover exists")
        self.graph = graph
        # the blossom matcher lists each pair once, so both ends are marked matched here
        self.matching = Blossom(graph).matching()
        self.matched = {n for pair in self.matching.items() for n in pair}
        self.cover = self._build()

    def _build(self) -> list[tuple[str, str]]:
        chosen: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for u, v in self.matching.items():
            key = (min(u, v), max(u, v))
            if key not in seen:
                seen.add(key)
                chosen.append(key)
        for node in self.graph.nodes():
            if node not in self.matched:
                partner = min(self.graph.neighbors(node))
                key = (min(node, partner), max(node, partner))
                if key not in seen:
                    seen.add(key)
                    chosen.append(key)
        return chosen

    def matching_size(self) -> int:
        return len(self.matching)

    def covers_every_node(self) -> bool:
        touched = {n for edge in self.cover for n in edge}
        return touched == set(self.graph.nodes())

    def gallai_holds(self) -> bool:
        return len(self.cover) == self.graph.node_count() - self.matching_size()

    def note(self) -> str:
        return (
            f"{len(self.cover)} edge(s) cover {self.graph.node_count()} node(s) from a "
            f"matching of {self.matching_size()}; Gallai's identity "
            f"{'holds' if self.gallai_holds() else 'fails'}"
        )
