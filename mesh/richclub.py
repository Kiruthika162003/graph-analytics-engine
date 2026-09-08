"""Rich club: do the best-connected nodes form a club among themselves?

The rich-club coefficient at level k looks only at the nodes whose
degree exceeds k, the rich ones, and asks what fraction of the possible
edges among them actually exist. A coefficient rising toward one as k
grows means the hubs are densely wired to each other, an elite that
talks mostly to itself, the pattern of airline hubs and of scientific
collaboration where the most prolific authors coauthor together. The
raw coefficient has a known trap: high-degree nodes have many edges, so
they connect to each other more than low-degree nodes would even in a
graph with no club structure at all, purely because they have more
edges to spend. The honest measure divides the raw coefficient by the
same coefficient measured on a randomized graph that keeps every node's
degree exactly but rewires who connects to whom, made by repeatedly
swapping the endpoints of two random edges. A normalized coefficient
above one is a rich club beyond what the degrees alone would produce;
near one is no club, just degree; below one is hubs that avoid each
other, as in some technological networks where hubs serve disjoint
regions. The engine computes the raw coefficient for every degree level,
builds the degree-preserving randomization with a seeded swap count of
several times the edge count, computes the same curve on it, and
reports the normalized coefficient at a chosen level. It refuses a level
that leaves fewer than two rich nodes, since no edge among them is
possible, and reports the level where the normalized curve peaks,
because the peak is where the club is most pronounced and a curve that
never rises above one is a graph whose hubs are no closer than chance.
"""

from __future__ import annotations

import random

from mesh.errors import Invalid
from mesh.graph import Graph


class RichClub:
    def __init__(self, graph: Graph, swaps_per_edge: int = 10, seed: int = 0) -> None:
        if graph.directed:
            raise Invalid("the rich-club coefficient here is for undirected graphs")
        if graph.edge_count() < 2:
            raise Invalid("a rich club needs at least two edges to rewire")
        self.graph = graph
        self.degree = {n: graph.degree(n) for n in graph.nodes()}
        self.baseline = self._randomized(swaps_per_edge, seed)

    @staticmethod
    def coefficient(graph: Graph, k: int) -> float:
        rich = [n for n in graph.nodes() if graph.degree(n) > k]
        if len(rich) < 2:
            raise Invalid(f"fewer than two nodes have degree above {k}; no club is possible")
        rich_set = set(rich)
        among = sum(1 for u, v, _w in graph.edges() if u in rich_set and v in rich_set)
        possible = len(rich) * (len(rich) - 1) / 2
        return among / possible

    def _randomized(self, swaps_per_edge: int, seed: int) -> Graph:
        # degree-preserving rewiring: swap the ends of two random edges
        rng = random.Random(seed)
        edges = [(u, v) for u, v, _w in self.graph.edges()]
        edge_set = {frozenset(e) for e in edges}
        for _ in range(swaps_per_edge * len(edges)):
            i, j = rng.sample(range(len(edges)), 2)
            a, b = edges[i]
            c, d = edges[j]
            if len({a, b, c, d}) < 4:
                continue  # a shared endpoint would create a loop or double edge
            new1, new2 = frozenset((a, d)), frozenset((c, b))
            if new1 in edge_set or new2 in edge_set:
                continue
            edge_set.discard(frozenset((a, b)))
            edge_set.discard(frozenset((c, d)))
            edge_set.add(new1)
            edge_set.add(new2)
            edges[i], edges[j] = (a, d), (c, b)
        g = Graph()
        for n in self.graph.nodes():
            g.add_node(n)
        for u, v in edges:
            g.add_edge(u, v)
        return g

    def levels(self) -> list[int]:
        top = max(self.degree.values())
        return [k for k in range(top) if sum(1 for d in self.degree.values() if d > k) >= 2]

    def normalized(self, k: int) -> float:
        raw = self.coefficient(self.graph, k)
        base = self.coefficient(self.baseline, k)
        return raw / base if base > 0 else float("inf") if raw > 0 else 1.0

    def peak_level(self) -> int:
        return max(self.levels(), key=lambda k: (self.normalized(k), -k))

    def note(self) -> str:
        k = self.peak_level()
        value = self.normalized(k)
        verdict = "a club beyond degree" if value > 1.05 else \
            "hubs no closer than chance" if value > 0.95 else "hubs that avoid each other"
        return f"normalized rich-club peaks at {value:.2f} for degree above {k}: {verdict}"
