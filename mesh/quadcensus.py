"""Quad census: every four nodes sorted into the six connected shapes or left uncounted.

Three-node shapes are the triad census and the graphlet orbits; four
nodes are the next rung, and on an undirected graph the connected
induced shapes on four nodes are exactly six: the path, the claw
with one center and three leaves, the four-cycle, the paw that is a
triangle with a pendant, the diamond that is a four-cycle with one
chord, and the complete graph on four. Each is told from the others
by its edge count and its degree sequence: three edges split the
path with degrees 1,1,2,2 from the claw with 1,1,1,3; four edges
split the cycle with 2,2,2,2 from the paw with 1,2,2,3; five edges
are the diamond and six the complete graph. Fewer than three edges,
or three edges as a triangle with an isolated node, is not
connected and is counted only in the disconnected tally. The engine
walks every four-subset, which is fine to a few dozen nodes, reads
the shape, and reports the counts. Identities pin the arithmetic:
the counts sum to n choose 4, a complete graph on n nodes has n
choose 4 complete quads and nothing else, a cycle on n at least five
has exactly n paths of four, a star on n leaves has n choose 3 claws,
and the total triangle count equals the triangles inside paws,
diamonds, and complete quads weighted by how many each holds, divided
by the number of quads a triangle sits in, which the module checks
against a direct triangle count. A directed graph is refused.
"""

from __future__ import annotations

from itertools import combinations
from math import comb

from mesh.errors import Invalid
from mesh.graph import Graph

SHAPES = ["path", "claw", "cycle", "paw", "diamond", "complete"]


class QuadCensus:
    def __init__(self, graph: Graph) -> None:
        if graph.directed:
            raise Invalid("the quad census reads undirected shapes")
        if graph.node_count() > 40:
            raise Invalid("the census walks every four-subset; keep it to forty nodes")
        self.graph = graph
        self.counts = dict.fromkeys(SHAPES, 0)
        self.disconnected = 0
        for quad in combinations(graph.nodes(), 4):
            shape = self.classify(quad)
            if shape is None:
                self.disconnected += 1
            else:
                self.counts[shape] += 1

    def classify(self, quad: tuple[str, ...]) -> str | None:
        edges = [(a, b) for a, b in combinations(quad, 2) if self.graph.has_edge(a, b)]
        degrees = sorted(sum(1 for e in edges if n in e) for n in quad)
        count = len(edges)
        if count == 6:
            return "complete"
        if count == 5:
            return "diamond"
        if count == 4:
            return "cycle" if degrees == [2, 2, 2, 2] else "paw"
        if count == 3:
            if degrees == [1, 1, 2, 2]:
                return "path"
            if degrees == [1, 1, 1, 3]:
                return "claw"
            return None  # a triangle beside an isolated node
        return None

    def total(self) -> int:
        return sum(self.counts.values()) + self.disconnected

    def sums_to_choose_four(self) -> bool:
        return self.total() == comb(self.graph.node_count(), 4)

    def triangles_by_search(self) -> int:
        has = self.graph.has_edge
        triples = combinations(self.graph.nodes(), 3)
        return sum(1 for a, b, c in triples if has(a, b) and has(b, c) and has(a, c))

    def triangles_from_quads(self) -> float:
        # each triangle lies in n - 3 quads; a paw holds one, a diamond two, K4 four
        n = self.graph.node_count()
        if n < 4:
            return float(self.triangles_by_search())
        weighted = self.counts["paw"] + 2 * self.counts["diamond"] + 4 * self.counts["complete"]
        weighted += self._lonely_triangles()
        return weighted / (n - 3)

    def _lonely_triangles(self) -> int:
        # triangles beside an isolated fourth node were tallied as disconnected
        has = self.graph.has_edge
        lonely = 0
        for quad in combinations(self.graph.nodes(), 4):
            edges = sum(1 for a, b in combinations(quad, 2) if has(a, b))
            if edges == 3 and self.classify(quad) is None:
                lonely += 1
        return lonely

    def triangle_identity_holds(self) -> bool:
        return abs(self.triangles_from_quads() - self.triangles_by_search()) < 1e-9

    def note(self) -> str:
        present = ", ".join(f"{s} x{c}" for s, c in self.counts.items() if c)
        return (
            f"{self.total()} quad(s): {present or 'none connected'}; "
            f"{self.disconnected} not connected"
        )
