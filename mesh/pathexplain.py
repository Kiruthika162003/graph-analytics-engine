"""Path explanation: why this route, what each step costs, and how much slack each edge has.

A shortest path is an answer; an explanation says why it is the
answer. For a route from source to sink this module reports each
step's weight and the running total, and for each edge on the route
the length of the best route that avoids it, so the difference is
the edge's slack: zero slack means an equally good route exists
without it, a large slack means the edge is doing the work, and an
infinite slack means the edge is the only way. It also reports the
runner-up, the shortest route that differs from the chosen one in at
least one edge, found by trying each avoided edge in turn, which is
the simplest of Yen's ideas and enough to say how close the second
choice was. Distances come from a heap Dijkstra written here with
parent pointers, refusing negative weights, and the avoided-edge
routes come from running it on a copy of the graph with the edge
removed, which is slower than a clever reoptimisation but has no
special cases to get wrong. The tests hold the facts an explanation
must respect: the steps sum to the distance, a bridge on the route
has infinite slack, an edge with a parallel route of equal length
has zero slack, the runner-up is never shorter than the route, and a
sink the source cannot reach is refused as unreachable. Directed
graphs are followed along arcs.
"""

from __future__ import annotations

import heapq
from itertools import pairwise
from math import inf

from mesh.errors import Invalid, Unreachable
from mesh.graph import Graph


def _dijkstra(graph: Graph, source: str) -> tuple[dict[str, float], dict[str, str | None]]:
    dist: dict[str, float] = {source: 0.0}
    parent: dict[str, str | None] = {source: None}
    heap = [(0.0, source)]
    while heap:
        d, node = heapq.heappop(heap)
        if d > dist.get(node, inf):
            continue
        for other, w in graph.neighbors(node).items():
            if w < 0:
                raise Invalid(f"edge {node}-{other} has a negative weight")
            cand = d + w
            if cand < dist.get(other, inf):
                dist[other] = cand
                parent[other] = node
                heapq.heappush(heap, (cand, other))
    return dist, parent


def _without(graph: Graph, a: str, b: str) -> Graph:
    g = Graph(directed=graph.directed)
    for n in graph.nodes():
        g.add_node(n)
    for u, v, w in graph.edges():
        if (u, v) == (a, b) or (not graph.directed and (u, v) == (b, a)):
            continue
        g.add_edge(u, v, w)
    return g


class PathExplanation:
    def __init__(self, graph: Graph, source: str, sink: str) -> None:
        for name in (source, sink):
            if name not in graph.nodes():
                raise Invalid(f"'{name}' is not a node of the graph")
        self.graph = graph
        self.source = source
        self.sink = sink
        dist, parent = _dijkstra(graph, source)
        if sink not in dist:
            raise Unreachable(f"'{sink}' cannot be reached from '{source}'")
        self.distance = dist[sink]
        trail = [sink]
        while parent[trail[-1]] is not None:
            trail.append(parent[trail[-1]])  # type: ignore[arg-type]
        self.route = list(reversed(trail))

    def steps(self) -> list[tuple[str, str, float, float]]:
        out = []
        running = 0.0
        for a, b in pairwise(self.route):
            w = self.graph.weight(a, b)
            running += w
            out.append((a, b, w, running))
        return out

    def slack(self) -> dict[tuple[str, str], float]:
        out: dict[tuple[str, str], float] = {}
        for a, b in pairwise(self.route):
            dist, _parent = _dijkstra(_without(self.graph, a, b), self.source)
            without = dist.get(self.sink, inf)
            out[(a, b)] = without - self.distance
        return out

    def runner_up(self) -> tuple[list[str], float] | None:
        best: tuple[list[str], float] | None = None
        for a, b in pairwise(self.route):
            trimmed = _without(self.graph, a, b)
            dist, parent = _dijkstra(trimmed, self.source)
            if self.sink not in dist:
                continue
            trail = [self.sink]
            while parent[trail[-1]] is not None:
                trail.append(parent[trail[-1]])  # type: ignore[arg-type]
            candidate = (list(reversed(trail)), dist[self.sink])
            if best is None or candidate[1] < best[1]:
                best = candidate
        return best

    def steps_sum_to_distance(self) -> bool:
        return abs(sum(w for _a, _b, w, _r in self.steps()) - self.distance) < 1e-9

    def note(self) -> str:
        route = " > ".join(self.route)
        slack = self.slack()
        tight = [f"{a}-{b}" for (a, b), s in slack.items() if s == inf]
        runner = self.runner_up()
        second = f"runner-up {runner[1]:g}" if runner else "no other route"
        return (
            f"{route} at {self.distance:g}; {len(tight)} edge(s) with no alternative "
            f"({', '.join(tight) if tight else 'none'}); {second}"
        )
