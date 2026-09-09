"""Layouts: positions for the nodes, on a circle, by force, or in layers for a DAG.

A drawing needs coordinates, and three ways of choosing them cover
most graphs. The circle layout spaces the nodes evenly around a ring
in sorted order, which never overlaps and never flatters. The spring
layout of Fruchterman and Reingold treats edges as springs that pull
and nodes as charges that push, moves every node a little along the
net force for a fixed number of rounds with a cooling step, and
settles into a picture where connected nodes sit near each other; it
starts from seeded random positions so the result is reproducible.
The layered layout takes a directed acyclic graph, assigns each node
the layer of its longest path from a source, orders each layer by
the mean position of its predecessors to reduce crossings in the
manner of Sugiyama's barycenter step, and spaces layers vertically,
which is the shape of a build graph or a dependency chart. All three
return positions in the unit square, which the renderer scales. The
tests hold what a layout must satisfy: every node gets a position,
positions stay inside the square, the circle keeps equal spacing, the
spring pulls two cliques joined by a bridge into two separate
clusters with the bridge spanning them, and the layered layout puts
every arc's head strictly below its tail. A layered layout of a graph
with a cycle is refused, since no layer order exists.
"""

from __future__ import annotations

import random
from math import cos, pi, sin, sqrt

from mesh.errors import Cyclic, Invalid
from mesh.graph import Graph

Point = tuple[float, float]


class Layout:
    def __init__(self, graph: Graph) -> None:
        self.graph = graph
        self.nodes = graph.nodes()

    def circle(self) -> dict[str, Point]:
        n = len(self.nodes)
        out: dict[str, Point] = {}
        for i, node in enumerate(sorted(self.nodes)):
            angle = 2 * pi * i / n if n else 0.0
            out[node] = (0.5 + 0.45 * cos(angle), 0.5 + 0.45 * sin(angle))
        return out

    def spring(self, rounds: int = 200, seed: int = 0) -> dict[str, Point]:
        if rounds < 0:
            raise Invalid("rounds cannot be negative")
        rng = random.Random(seed)
        pos = {n: (rng.random(), rng.random()) for n in self.nodes}
        n = len(self.nodes)
        if n < 2:
            return dict.fromkeys(self.nodes, (0.5, 0.5))
        k = sqrt(1.0 / n)
        temperature = 0.1
        for _ in range(rounds):
            force = dict.fromkeys(self.nodes, (0.0, 0.0))
            for i, a in enumerate(self.nodes):
                for b in self.nodes[i + 1 :]:
                    dx, dy = pos[a][0] - pos[b][0], pos[a][1] - pos[b][1]
                    dist = max(sqrt(dx * dx + dy * dy), 1e-6)
                    push = k * k / dist
                    fx, fy = dx / dist * push, dy / dist * push
                    force[a] = (force[a][0] + fx, force[a][1] + fy)
                    force[b] = (force[b][0] - fx, force[b][1] - fy)
            for u, v, _w in self.graph.edges():
                dx, dy = pos[u][0] - pos[v][0], pos[u][1] - pos[v][1]
                dist = max(sqrt(dx * dx + dy * dy), 1e-6)
                pull = dist * dist / k
                fx, fy = dx / dist * pull, dy / dist * pull
                force[u] = (force[u][0] - fx, force[u][1] - fy)
                force[v] = (force[v][0] + fx, force[v][1] + fy)
            for node in self.nodes:
                fx, fy = force[node]
                size = max(sqrt(fx * fx + fy * fy), 1e-6)
                step = min(size, temperature)
                x = min(1.0, max(0.0, pos[node][0] + fx / size * step))
                y = min(1.0, max(0.0, pos[node][1] + fy / size * step))
                pos[node] = (x, y)
            temperature *= 0.97
        return pos

    def layered(self) -> dict[str, Point]:
        if not self.graph.directed:
            raise Invalid("a layered layout needs a directed acyclic graph")
        indegree = {n: self.graph.in_degree(n) for n in self.nodes}
        layer = dict.fromkeys(self.nodes, 0)
        ready = sorted(n for n in self.nodes if indegree[n] == 0)
        seen = 0
        while ready:
            node = ready.pop(0)
            seen += 1
            for other in self.graph.neighbors(node):
                layer[other] = max(layer[other], layer[node] + 1)
                indegree[other] -= 1
                if indegree[other] == 0:
                    ready.append(other)
                    ready.sort()
        if seen != len(self.nodes):
            raise Cyclic("a cycle leaves some nodes without a layer")
        depth = max(layer.values(), default=0)
        rows: dict[int, list[str]] = {}
        for node in self.nodes:
            rows.setdefault(layer[node], []).append(node)
        pos: dict[str, Point] = {}
        for level in range(depth + 1):
            members = rows.get(level, [])
            if level == 0:
                members.sort()
            else:
                # the barycenter step: order by the mean x of predecessors
                def key(node: str) -> tuple[float, str]:
                    preds = [p for p in self.nodes if self.graph.has_edge(p, node) and p in pos]
                    mean = sum(pos[p][0] for p in preds) / len(preds) if preds else 0.5
                    return (mean, node)

                members.sort(key=key)
            for i, node in enumerate(members):
                x = (i + 1) / (len(members) + 1)
                y = level / depth if depth else 0.5
                pos[node] = (x, y)
        return pos

    @staticmethod
    def inside_square(pos: dict[str, Point]) -> bool:
        return all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in pos.values())

    def note(self, pos: dict[str, Point], method: str) -> str:
        inside = self.inside_square(pos)
        return f"{method} layout of {len(pos)} node(s), all inside the unit square: {inside}"
