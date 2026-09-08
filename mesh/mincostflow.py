"""Min-cost flow: push the most you can, along the cheapest routes first.

Maximum flow asks how much can be sent; minimum-cost flow asks how to send
a given amount as cheaply as possible when every edge charges a price per
unit. Shipping goods through a network of roads with tolls, routing
traffic where each link has a latency, assigning workers to jobs with
costs, all are min-cost flow, and the assignment problem is the special
case with unit capacities. The successive shortest path method solves it
by a simple loop: find the cheapest path from source to sink in the
residual graph, where the cost of a reverse edge is the negative of the
forward edge it cancels, push as much flow as that path can carry, and
repeat until the required amount is sent or no path remains. Each
augmentation is along the cheapest available route, and the argument that
this stays optimal is that a min-cost flow of value f, augmented along a
shortest path in its residual graph, is a min-cost flow of value f plus
the pushed amount; the reverse edges are what let a later cheaper path
undo an earlier routing. Because reverse edges carry negative costs, the
shortest path search must tolerate them, so the engine uses Bellman-Ford
rather than Dijkstra; a negative cycle in the residual graph cannot arise
from a min-cost starting state, which is what keeps Bellman-Ford's
detection quiet. The solver takes capacities and per-unit costs, sends up
to a requested amount, returns the flow sent and its total cost, refuses a
negative capacity and a request that exceeds what the network can carry,
and reports the cost per unit against the cheapest single source-to-sink
path, because the gap between them is the price of the network's
bottlenecks forcing flow onto dearer routes.
"""

from __future__ import annotations

import math
from itertools import pairwise

from mesh.errors import Invalid, Missing


class MinCostFlow:
    def __init__(self) -> None:
        self._nodes: set[str] = set()
        # residual capacity and per-unit cost on every directed pair
        self._cap: dict[str, dict[str, float]] = {}
        self._cost: dict[str, dict[str, float]] = {}
        self.sent = 0.0
        self.total_cost = 0.0

    def add_edge(self, u: str, v: str, capacity: float, cost: float) -> None:
        if capacity < 0:
            raise Invalid(f"edge {u}->{v} has negative capacity {capacity}")
        for n in (u, v):
            self._nodes.add(n)
            self._cap.setdefault(n, {})
            self._cost.setdefault(n, {})
        self._cap[u][v] = self._cap[u].get(v, 0.0) + capacity
        self._cost[u][v] = cost
        self._cap[v].setdefault(u, 0.0)
        self._cost[v][u] = -cost  # the reverse edge refunds the cost

    def _cheapest_path(self, source: str, sink: str) -> list[str] | None:
        # Bellman-Ford over the residual graph; reverse edges carry negative cost
        dist = dict.fromkeys(self._nodes, math.inf)
        parent: dict[str, str | None] = {source: None}
        dist[source] = 0.0
        for _ in range(len(self._nodes) - 1):
            changed = False
            for u in self._nodes:
                if dist[u] == math.inf:
                    continue
                for v, cap in self._cap[u].items():
                    if cap > 0 and dist[u] + self._cost[u][v] < dist[v]:
                        dist[v] = dist[u] + self._cost[u][v]
                        parent[v] = u
                        changed = True
            if not changed:
                break
        if dist[sink] == math.inf:
            return None
        path = [sink]
        while parent[path[-1]] is not None:
            path.append(parent[path[-1]])  # type: ignore[arg-type]
        path.reverse()
        return path

    def send(self, source: str, sink: str, amount: float) -> float:
        for n in (source, sink):
            if n not in self._nodes:
                raise Missing(f"'{n}' is not in the network")
        remaining = amount
        while remaining > 0:
            path = self._cheapest_path(source, sink)
            if path is None:
                raise Invalid(
                    f"the network can carry only {self.sent} of the requested "
                    f"{amount}; the rest has no route left"
                )
            bottleneck = min(self._cap[u][v] for u, v in pairwise(path))
            push = min(bottleneck, remaining)
            for u, v in pairwise(path):
                self._cap[u][v] -= push
                self._cap[v][u] += push
                self.total_cost += push * self._cost[u][v]
            self.sent += push
            remaining -= push
        return self.total_cost

    def cheapest_single_path_cost(self, source: str, sink: str) -> float:
        path = self._cheapest_path(source, sink)
        if path is None:
            raise Invalid("no path remains")
        return sum(self._cost[u][v] for u, v in pairwise(path))

    def note(self) -> str:
        per_unit = self.total_cost / self.sent if self.sent else 0.0
        return (
            f"sent {self.sent} at total cost {self.total_cost}, {per_unit:.2f} per "
            "unit; the gap above the cheapest single path is the price of "
            "bottlenecks forcing flow onto dearer routes"
        )
