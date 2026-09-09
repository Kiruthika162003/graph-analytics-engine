"""Graph summary: one pass over a graph that reports the readings a first look wants.

Before any specific question, a reader wants the shape of a graph in
a few lines: how big, how dense, how many pieces, how far across, how
clustered, how centralised, and which recognised class it falls in if
any. This module gathers those from the engine's own modules and
prints them as one report, which is what the command line and the
examples use for a first look. Size is nodes and edges; density is
edges over the possible pairs; the degree line gives the minimum,
average, and maximum; pieces come from a breadth-first sweep; the
diameter and radius come from the center module when the graph is
connected and are reported as absent otherwise; the clustering line
reports the triangle count and the transitivity, three times
triangles over connected triples, from the graphlet orbits; and the
class line runs the tree, chordal, split, cograph, and threshold
recognisers and lists every one that says yes, since a graph can be
several at once. The summary never raises on a shape it cannot read:
a directed graph gets a shorter report that skips the undirected
readings and says why, and an empty graph reports its emptiness. The
identities the summary carries are checked in the tests, such as
transitivity being one on a complete graph and zero on a tree, and
the class line naming exactly the classes a star belongs to.
"""

from __future__ import annotations

from collections import deque

from mesh.chordal import Chordal
from mesh.cograph import Cograph
from mesh.graph import Graph
from mesh.graphcenter import GraphCenter
from mesh.graphlets import Graphlets
from mesh.splitgraph import SplitGraph
from mesh.thresholdgraph import ThresholdGraph


class GraphSummary:
    def __init__(self, graph: Graph) -> None:
        self.graph = graph
        self.n = graph.node_count()
        self.m = graph.edge_count()

    def density(self) -> float:
        if self.n < 2:
            return 0.0
        pairs = self.n * (self.n - 1)
        if not self.graph.directed:
            pairs //= 2
        return self.m / pairs

    def degrees(self) -> tuple[int, float, int]:
        if self.n == 0:
            return 0, 0.0, 0
        ds = [self.graph.degree(n) for n in self.graph.nodes()]
        return min(ds), sum(ds) / len(ds), max(ds)

    def pieces(self) -> int:
        seen: set[str] = set()
        count = 0
        for start in self.graph.nodes():
            if start in seen:
                continue
            count += 1
            seen.add(start)
            queue = deque([start])
            while queue:
                node = queue.popleft()
                for other in self.graph.neighbors(node):
                    if other not in seen:
                        seen.add(other)
                        queue.append(other)
        return count

    def transitivity(self) -> float:
        gl = Graphlets(self.graph)
        triples = gl.paths + 3 * gl.triangles
        return 3 * gl.triangles / triples if triples else 0.0

    def classes(self) -> list[str]:
        found = []
        if self.n and self.m == self.n - 1 and self.pieces() == 1:
            found.append("tree")
        if Chordal(self.graph).is_chordal:
            found.append("chordal")
        if SplitGraph(self.graph).is_split:
            found.append("split")
        if Cograph(self.graph).is_cograph:
            found.append("cograph")
        if ThresholdGraph(self.graph).is_threshold:
            found.append("threshold")
        return found

    def lines(self) -> list[str]:
        if self.n == 0:
            return ["empty graph: no nodes"]
        kind = "directed" if self.graph.directed else "undirected"
        low, avg, high = self.degrees()
        size = f"{self.n} node(s) and {self.m} edge(s)"
        out = [
            f"{kind} graph with {size}, density {self.density():.3f}",
            f"degrees from {low} to {high}, average {avg:.2f}; {self.pieces()} piece(s)",
        ]
        if self.graph.directed:
            out.append("distance, clustering, and class readings are for undirected graphs")
            return out
        center = GraphCenter(self.graph)
        if self.pieces() == 1:
            out.append(
                f"radius {center.radius():g}, diameter {center.diameter():g}, "
                f"center {center.center()}"
            )
        else:
            out.append("radius and diameter absent: the graph is not connected")
        gl = Graphlets(self.graph)
        out.append(f"{gl.triangles} triangle(s), transitivity {self.transitivity():.3f}")
        found = self.classes()
        out.append("classes: " + (", ".join(found) if found else "none recognised"))
        return out

    def note(self) -> str:
        return "\n".join(self.lines())
