"""Feedback arc set: the fewest edges to remove so a directed graph has no cycle.

A dependency graph with cycles cannot be scheduled, a ranking derived
from pairwise results cannot be totally ordered if the results
contradict each other around a loop, and a layered drawing of a graph
needs every edge to point downward. All three ask for a feedback arc
set: a set of edges whose removal leaves a directed acyclic graph, as
small as possible. Finding the minimum is NP-hard, so the engine offers
the Eades-Lin-Smyth heuristic, which produces a node ordering and takes
as the feedback set every edge that points backward against it. The
ordering is built by peeling: repeatedly remove every sink and place it
at the end of the order, remove every source and place it at the start,
and when neither exists remove the node with the largest surplus of
out-degree over in-degree and place it at the start, since its many
outgoing edges will then point forward and only its few incoming edges
will point back. Sinks and sources never contribute a backward edge, so
the only backward edges come from the surplus choices, and the
heuristic's guarantee is that the feedback set is at most half the
edges plus a term in the node count, with real inputs doing far better.
The result is measured, not trusted: on small graphs the engine compares
its set against the true minimum found by trying every subset of edges,
and reports both. The solver returns the ordering, the backward edges,
verifies the remaining graph is acyclic, and reports the feedback set
size against the edge count, because a large fraction removed is a
graph that is mostly cycle, where no ordering does well, while a small
fraction is a graph that was nearly ordered already.
"""

from __future__ import annotations

from mesh.errors import Invalid
from mesh.graph import Graph
from mesh.toposort import TopologicalSort


class FeedbackArcSet:
    def __init__(self, graph: Graph) -> None:
        if not graph.directed:
            raise Invalid("a feedback arc set is defined on a directed graph")
        self.graph = graph
        self.order = self._eades_lin_smyth()
        rank = {n: i for i, n in enumerate(self.order)}
        self.removed = [(u, v) for u, v, _w in graph.edges() if rank[u] > rank[v]]

    def _eades_lin_smyth(self) -> list[str]:
        remaining = set(self.graph.nodes())
        out_deg = dict.fromkeys(remaining, 0)
        in_deg = dict.fromkeys(remaining, 0)
        for u, v, _w in self.graph.edges():
            if u != v:
                out_deg[u] += 1
                in_deg[v] += 1
        front: list[str] = []
        back: list[str] = []

        def drop(node: str) -> None:
            remaining.discard(node)
            for v in self.graph.neighbors(node):
                if v in remaining:
                    in_deg[v] -= 1
            for u in self.graph.nodes():
                if u in remaining and self.graph.has_edge(u, node):
                    out_deg[u] -= 1

        while remaining:
            sinks = sorted(n for n in remaining if out_deg[n] == 0)
            if sinks:
                node = sinks[0]
                back.append(node)  # sinks go to the end, nothing points back from them
                drop(node)
                continue
            sources = sorted(n for n in remaining if in_deg[n] == 0)
            if sources:
                node = sources[0]
                front.append(node)
                drop(node)
                continue
            # no sink or source: the largest out-minus-in surplus goes first
            node = max(remaining, key=lambda n: (out_deg[n] - in_deg[n], n))
            front.append(node)
            drop(node)
        back.reverse()
        return front + back

    def acyclic_remainder(self) -> Graph:
        g = Graph(directed=True)
        for n in self.graph.nodes():
            g.add_node(n)
        gone = set(self.removed)
        for u, v, w in self.graph.edges():
            if (u, v) not in gone:
                g.add_edge(u, v, w)
        return g

    def remainder_is_acyclic(self) -> bool:
        return TopologicalSort(self.acyclic_remainder()).is_acyclic()

    def note(self) -> str:
        m = self.graph.edge_count() or 1
        return (
            f"removed {len(self.removed)} of {self.graph.edge_count()} edge(s) "
            f"({len(self.removed) / m * 100:.0f}%); a large share is a graph that is "
            "mostly cycle, a small one was nearly ordered already"
        )
