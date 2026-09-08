"""Subgraph matching: find every copy of a small pattern inside a large graph.

Counting how often a small motif appears, a triangle, a four-cycle, a
feed-forward loop of three nodes, is the standard way to characterize a
network beyond its degrees, and answering whether one specific pattern
appears at all is the subgraph isomorphism problem, NP-complete in
general but fast in practice for small patterns because the pattern's
constraints prune the search hard. The engine maps pattern nodes to
graph nodes one at a time by backtracking. A candidate for the next
pattern node must have degree at least the pattern node's, must not be
already used, and must be adjacent to the images of exactly those
already-mapped pattern nodes that the pattern node is adjacent to;
under induced matching it must also be non-adjacent where the pattern
is non-adjacent, so an induced triangle count excludes triangles inside
a four-clique's chords, while the non-induced count includes them. The
pattern's own symmetries mean the same copy is found once per
automorphism, a triangle six times, so the engine divides by the
automorphism count, computed by matching the pattern against itself, to
report distinct copies, and states that division rather than leaving
the raw count to be misread. Ordering the pattern nodes so that each
after the first is adjacent to an earlier one keeps the search
connected and the candidate sets small. The matcher returns the
distinct copy count, one witness mapping, whether the pattern occurs,
and refuses a pattern larger than the graph or one mixing directedness
with it. It reports the copies found against the raw mappings explored,
because a raw count far above the copies is a highly symmetric pattern
where most of the search was rediscovering the same copy.
"""

from __future__ import annotations

from mesh.errors import Invalid
from mesh.graph import Graph


class SubgraphMatch:
    def __init__(self, graph: Graph, pattern: Graph, induced: bool = False) -> None:
        if graph.directed != pattern.directed:
            raise Invalid("the pattern and the graph must both be directed or both not")
        if pattern.node_count() == 0 or pattern.node_count() > graph.node_count():
            raise Invalid("the pattern must be non-empty and no larger than the graph")
        self.graph = graph
        self.pattern = pattern
        self.induced = induced
        self.order = self._connected_order(pattern)
        self.mappings = 0
        self.witness: dict[str, str] | None = None
        self._search(graph, {}, set(), count_all=True)
        self.raw = self.mappings
        self.automorphisms = self._count_automorphisms()
        self.copies = self.raw // self.automorphisms if self.automorphisms else 0

    @staticmethod
    def _connected_order(pattern: Graph) -> list[str]:
        nodes = sorted(pattern.nodes(), key=lambda n: (-pattern.degree(n), n))
        order = [nodes[0]]
        placed = {nodes[0]}
        while len(order) < len(nodes):
            nxt = next(
                (n for n in nodes if n not in placed and any(
                    pattern.has_edge(n, p) or pattern.has_edge(p, n) for p in order
                )),
                None,
            )
            if nxt is None:
                nxt = next(n for n in nodes if n not in placed)  # a disconnected piece
            order.append(nxt)
            placed.add(nxt)
        return order

    def _fits(self, target: Graph, mapping: dict[str, str], p: str, g: str) -> bool:
        if target.degree(g) < self.pattern.degree(p):
            return False
        for q, h in mapping.items():
            forward = self.pattern.has_edge(p, q)
            backward = self.pattern.has_edge(q, p)
            if forward and not target.has_edge(g, h):
                return False
            if backward and not target.has_edge(h, g):
                return False
            if self.induced:
                if not forward and target.has_edge(g, h):
                    return False
                if not backward and target.has_edge(h, g):
                    return False
        return True

    def _search(
        self, target: Graph, mapping: dict[str, str], used: set[str], count_all: bool
    ) -> bool:
        if len(mapping) == len(self.order):
            self.mappings += 1
            if self.witness is None:
                self.witness = dict(mapping)
            return True
        p = self.order[len(mapping)]
        found = False
        for g in target.nodes():
            if g in used or not self._fits(target, mapping, p, g):
                continue
            mapping[p] = g
            used.add(g)
            hit = self._search(target, mapping, used, count_all)
            del mapping[p]
            used.discard(g)
            if hit and not count_all:
                return True
            found = found or hit
        return found

    def _count_automorphisms(self) -> int:
        # match the pattern against itself; save and restore the graph search's
        # state so a self-mapping never masquerades as a witness in the graph
        saved_mappings, saved_witness = self.mappings, self.witness
        self.mappings = 0
        self._search(self.pattern, {}, set(), count_all=True)
        autos = self.mappings
        self.mappings, self.witness = saved_mappings, saved_witness
        return autos

    def occurs(self) -> bool:
        return self.copies > 0

    def note(self) -> str:
        return (
            f"{self.copies} distinct cop(ies) from {self.raw} raw mapping(s) over "
            f"{self.automorphisms} automorphism(s); raw far above copies is a symmetric "
            "pattern rediscovering itself"
        )
