"""Minimum cut: the exact edges whose removal severs source from sink, from a max flow.

Max-flow min-cut says the value of the maximum flow equals the capacity
of the minimum cut, but the theorem is about a number; the engineer
usually wants the cut itself, the specific edges to cut or the specific
links that form the bottleneck. Those edges fall straight out of a
finished maximum flow. In the residual graph left after the last
augmentation, run a search from the source along edges with residual
capacity remaining. The set of nodes reached is the source side of the
cut, everything else is the sink side, and the cut edges are the
original edges that run from the source side to the sink side. Every one
of those edges is saturated, carrying exactly its capacity, because if any
had residual capacity the search would have crossed it; and no edge runs
back with flow from sink side to source side, because that flow would
have opened a reverse residual edge for the search to cross. So the
capacity of those crossing edges is exactly the flow value, which is the
constructive half of the theorem: not only are the two numbers equal, here
are the edges that make them equal. There can be several minimum cuts of
the same capacity, and the residual search returns the one closest to the
source, the smallest source side; searching backward from the sink would
give the one closest to the sink. The extractor runs the flow, takes the
residual reachability, returns the source side and the crossing edges,
verifies their capacity sums to the flow value, and reports the cut size
in edges against the flow value, because a cut of a few high-capacity
edges is a network whose bottleneck is a handful of fat links while a cut
of many thin edges is a broad front with no single point to reinforce.
"""

from __future__ import annotations

from mesh.edmondskarp import EdmondsKarp
from mesh.graph import Graph


class MinCut:
    def __init__(self, graph: Graph, source: str, sink: str) -> None:
        self.graph = graph
        self.source = source
        self.sink = sink
        self._flow = EdmondsKarp(graph, source, sink)
        self.value = self._flow.value
        self.source_side = self._reachable_in_residual()
        self.edges = [
            (u, v, w)
            for u, v, w in graph.edges()
            if u in self.source_side and v not in self.source_side
        ]

    def _reachable_in_residual(self) -> set[str]:
        # everything the source still reaches along unsaturated residual edges
        seen = {self.source}
        stack = [self.source]
        while stack:
            u = stack.pop()
            for v, cap in self._flow.residual[u].items():
                if cap > 0 and v not in seen:
                    seen.add(v)
                    stack.append(v)
        return seen

    def capacity(self) -> float:
        return sum(w for _u, _v, w in self.edges)

    def matches_flow(self) -> bool:
        return self.capacity() == self.value

    def every_cut_edge_is_saturated(self) -> bool:
        return all(self._flow.flow_on(u, v) == w for u, v, w in self.edges)

    def note(self) -> str:
        return (
            f"cut of {len(self.edges)} edge(s) with capacity {self.capacity()} "
            f"against flow {self.value}; few fat edges is a bottleneck to reinforce, "
            "many thin ones is a broad front"
        )
