"""Flow decomposition: a maximum flow taken apart into the routes and loops that carry it.

A flow value is a single number, and a flow is a number on every arc,
but what a dispatcher wants is routes: this much along this path from
source to sink. The decomposition theorem says every flow is a sum of
at most m path flows and cycle flows, where m is the arc count, and the
proof is the algorithm. Take the net flow on every arc, keeping only
the positive direction of any antiparallel pair. Walk from the source
along arcs that still carry flow until the sink appears; the smallest
remaining flow on that walk is the route's amount, subtract it from
every arc of the walk, and at least one arc drops to zero, which is
why the count is bounded by m. When the source can no longer reach the
sink through carrying arcs, whatever remains is circulation, and a walk
from any carrying arc must revisit a node, which closes a cycle to
subtract the same way. The engine runs the Edmonds-Karp flow, takes it
apart into routes and cycles, and checks that the route amounts sum to
the flow value, that every route walks arcs of the graph from source to
sink, and that nothing is left on any arc at the end. The walk follows
the smallest-named carrying neighbor so the decomposition is
deterministic. A graph with no arc carrying flow decomposes to nothing.
"""

from __future__ import annotations

from itertools import pairwise

from mesh.edmondskarp import EdmondsKarp
from mesh.graph import Graph

Arc = tuple[str, str]


class FlowDecomposition:
    def __init__(self, graph: Graph, source: str, sink: str) -> None:
        self.graph = graph
        self.source = source
        self.sink = sink
        self.flow = EdmondsKarp(graph, source, sink)
        self.value = self.flow.value
        self.carrying = self._net_flows()
        self.paths: list[tuple[list[str], float]] = []
        self.cycles: list[tuple[list[str], float]] = []
        self._decompose()

    def _net_flows(self) -> dict[Arc, float]:
        # keep the positive direction only, so an antiparallel pair counts once
        net: dict[Arc, float] = {}
        for u, v, _w in self.graph.edges():
            f = self.flow.flow_on(u, v)
            if f > 0:
                net[(u, v)] = f
        return net

    def _out(self, node: str) -> list[str]:
        return sorted(v for (u, v), f in self.carrying.items() if u == node and f > 0)

    def _walk_to_sink(self) -> list[str] | None:
        # depth-first along carrying arcs, smallest name first, until the sink
        stack = [(self.source, [self.source])]
        seen = {self.source}
        while stack:
            node, trail = stack.pop()
            if node == self.sink:
                return trail
            for nxt in reversed(self._out(node)):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append((nxt, [*trail, nxt]))
        return None

    def _subtract(self, trail: list[str]) -> float:
        arcs = list(pairwise(trail))
        amount = min(self.carrying[a] for a in arcs)
        for a in arcs:
            self.carrying[a] -= amount
            if self.carrying[a] <= 1e-12:
                del self.carrying[a]
        return amount

    def _find_cycle(self) -> list[str] | None:
        start = min((u for (u, _v) in self.carrying), default=None)
        if start is None:
            return None
        trail = [start]
        position = {start: 0}
        node = start
        while True:
            nxt = self._out(node)[0]
            if nxt in position:
                return [*trail[position[nxt] :], nxt]
            position[nxt] = len(trail)
            trail.append(nxt)
            node = nxt

    def _decompose(self) -> None:
        while (trail := self._walk_to_sink()) is not None:
            self.paths.append((trail, self._subtract(trail)))
        while (loop := self._find_cycle()) is not None:
            self.cycles.append((loop, self._subtract(loop)))

    def route_total(self) -> float:
        return sum(amount for _t, amount in self.paths)

    def sums_to_value(self) -> bool:
        return abs(self.route_total() - self.value) < 1e-9

    def routes_walk_arcs(self) -> bool:
        for trail, _amount in self.paths:
            if trail[0] != self.source or trail[-1] != self.sink:
                return False
            if not all(self.graph.has_edge(a, b) for a, b in pairwise(trail)):
                return False
        return True

    def nothing_left(self) -> bool:
        return not self.carrying

    def note(self) -> str:
        return (
            f"flow {self.value} carried by {len(self.paths)} route(s) and "
            f"{len(self.cycles)} cycle(s); routes sum to {self.route_total()}"
        )
