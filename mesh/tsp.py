"""Travelling salesman: the exact tour by Held-Karp, and the double-tree tour within twice it.

Visiting every node once and returning home as cheaply as possible is
the problem every routing question reduces to, and on a complete
graph with metric weights two answers are worth having. The exact one
is the Held-Karp dynamic programme: for every subset of nodes
containing the start and every last node in it, the cheapest path
from the start through exactly that subset ending there, built by
adding one node at a time; the tour closes by returning to the start.
It runs in n squared times two to the n and is honest to about twelve
nodes. The approximate one is the double-tree tour: take a minimum
spanning tree, walk it in preorder, and shortcut repeated nodes,
which visits every node in the order the walk first meets them. A
tour is at least the tree, since dropping one edge of a tour leaves a
spanning path, and the walk is twice the tree, so with the triangle
inequality the shortcut tour is at most twice the optimum. The engine
checks completeness of the graph, builds the tree by Prim, produces
both tours, verifies each visits every node once and returns home,
and reports the ratio, which the tests hold below two on random
points in the plane and which is exactly one on a graph whose tree is
already a path around a line. A graph that is not complete is refused
with the missing pair named, and a directed graph is refused.
"""

from __future__ import annotations

from itertools import combinations
from math import inf

from mesh.errors import Invalid
from mesh.graph import Graph


class TravellingSalesman:
    def __init__(self, graph: Graph) -> None:
        if graph.directed:
            raise Invalid("the tours here are on an undirected complete graph")
        self.graph = graph
        self.nodes = graph.nodes()
        for a, b in combinations(self.nodes, 2):
            if not graph.has_edge(a, b):
                raise Invalid(f"no edge {a}-{b}; the salesman needs a complete graph")
        self.index = {n: i for i, n in enumerate(self.nodes)}

    def _w(self, a: str, b: str) -> float:
        return self.graph.weight(a, b)

    def tour_length(self, tour: list[str]) -> float:
        if not tour:
            return 0.0
        total = 0.0
        for i, node in enumerate(tour):
            total += self._w(node, tour[(i + 1) % len(tour)]) if len(tour) > 1 else 0.0
        return total

    def is_tour(self, tour: list[str]) -> bool:
        return sorted(tour) == sorted(self.nodes) and len(set(tour)) == len(tour)

    def exact(self) -> list[str]:
        n = len(self.nodes)
        if n > 12:
            raise Invalid("Held-Karp is exponential; keep it to twelve nodes")
        if n <= 1:
            return list(self.nodes)
        start = 0
        # best[mask][last]: cheapest path from start through mask ending at last
        best = [[inf] * n for _ in range(1 << n)]
        parent = [[-1] * n for _ in range(1 << n)]
        best[1 << start][start] = 0.0
        for mask in range(1 << n):
            if not mask & (1 << start):
                continue
            for last in range(n):
                if not mask & (1 << last) or best[mask][last] == inf:
                    continue
                for nxt in range(n):
                    if mask & (1 << nxt):
                        continue
                    cand = best[mask][last] + self._w(self.nodes[last], self.nodes[nxt])
                    new_mask = mask | (1 << nxt)
                    if cand < best[new_mask][nxt]:
                        best[new_mask][nxt] = cand
                        parent[new_mask][nxt] = last
        full = (1 << n) - 1
        home = self.nodes[start]
        # the tour closes from any node but the start, which has no edge to itself
        last = min(
            (i for i in range(n) if i != start),
            key=lambda i: best[full][i] + self._w(self.nodes[i], home),
        )
        tour: list[str] = []
        mask = full
        while last != -1:
            tour.append(self.nodes[last])
            prev = parent[mask][last]
            mask &= ~(1 << last)
            last = prev
        tour.reverse()
        return tour

    def _prim(self) -> dict[str, list[str]]:
        children: dict[str, list[str]] = {n: [] for n in self.nodes}
        if not self.nodes:
            return children
        inside = {self.nodes[0]}
        while len(inside) < len(self.nodes):
            best_edge = min(
                ((self._w(u, v), u, v) for u in inside for v in self.nodes if v not in inside),
                key=lambda t: (t[0], t[1], t[2]),
            )
            _w, u, v = best_edge
            children[u].append(v)
            inside.add(v)
        return children

    def double_tree(self) -> list[str]:
        children = self._prim()
        if not self.nodes:
            return []
        tour: list[str] = []
        stack = [self.nodes[0]]
        while stack:
            node = stack.pop()
            tour.append(node)
            stack.extend(reversed(children[node]))
        return tour

    def ratio(self) -> float:
        exact = self.tour_length(self.exact())
        return self.tour_length(self.double_tree()) / exact if exact else 1.0

    def note(self) -> str:
        exact = self.exact()
        approx = self.double_tree()
        return (
            f"exact tour {self.tour_length(exact):g} via {' > '.join(exact)}; double-tree "
            f"tour {self.tour_length(approx):g}, ratio {self.ratio():.3f}"
        )
