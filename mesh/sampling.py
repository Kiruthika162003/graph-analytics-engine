"""Graph sampling: a piece of a big graph drawn by nodes, by edges, by snowball, or by walking.

A graph too large to analyse whole is read through a sample, and the
sample's method decides which readings survive. Node sampling picks
nodes uniformly and keeps the edges among them, which preserves
nothing about degrees in a sparse graph, since most of a sampled
node's neighbors are gone. Edge sampling picks edges uniformly and
keeps their endpoints, which favours high-degree nodes in proportion
to their degree. Snowball sampling starts from a seed and takes
everything within a few hops, which preserves local structure exactly
and global structure not at all. Random walk sampling follows a
walker for a fixed number of steps and keeps the nodes it visits,
which in the long run visits a node in proportion to its degree on an
undirected graph, so a walk sample is biased toward hubs in a known
way that can be corrected by weighting each visit by one over the
degree. The engine offers all four with a seed, returns the induced
subgraph on the sampled nodes, and reports the degree bias of a
sample as the mean degree of its nodes in the original graph against
the graph's mean degree, so a reader sees which method skewed what.
The walk's stationary claim is checked directly: over many steps on
a connected undirected graph the visit frequency of each node tracks
its degree share, and the degree-corrected frequency tracks a uniform
share. A directed graph is walked along its arcs and may trap the
walker in a sink, which the walker escapes by restarting at a random
node, and the module counts those restarts.
"""

from __future__ import annotations

import random
from collections import Counter, deque

from mesh.errors import Invalid
from mesh.graph import Graph


class Sampler:
    def __init__(self, graph: Graph, seed: int = 0) -> None:
        self.graph = graph
        self.rng = random.Random(seed)
        self.nodes = graph.nodes()
        self.restarts = 0

    def induced(self, chosen: set[str]) -> Graph:
        g = Graph(directed=self.graph.directed)
        for n in self.nodes:
            if n in chosen:
                g.add_node(n)
        for u, v, w in self.graph.edges():
            if u in chosen and v in chosen:
                g.add_edge(u, v, w)
        return g

    def by_nodes(self, count: int) -> Graph:
        if not 0 <= count <= len(self.nodes):
            raise Invalid(f"cannot sample {count} of {len(self.nodes)} node(s)")
        return self.induced(set(self.rng.sample(self.nodes, count)))

    def by_edges(self, count: int) -> Graph:
        edges = [(u, v) for u, v, _w in self.graph.edges()]
        if not 0 <= count <= len(edges):
            raise Invalid(f"cannot sample {count} of {len(edges)} edge(s)")
        chosen = self.rng.sample(edges, count)
        return self.induced({n for e in chosen for n in e})

    def snowball(self, seed_node: str, hops: int) -> Graph:
        if seed_node not in self.nodes:
            raise Invalid(f"'{seed_node}' is not a node of the graph")
        if hops < 0:
            raise Invalid("hops cannot be negative")
        dist = {seed_node: 0}
        queue = deque([seed_node])
        while queue:
            node = queue.popleft()
            if dist[node] == hops:
                continue
            for other in self.graph.neighbors(node):
                if other not in dist:
                    dist[other] = dist[node] + 1
                    queue.append(other)
        return self.induced(set(dist))

    def walk(self, steps: int, start: str | None = None) -> list[str]:
        if steps < 0:
            raise Invalid("steps cannot be negative")
        if not self.nodes:
            return []
        node = start if start is not None else self.rng.choice(self.nodes)
        if node not in self.nodes:
            raise Invalid(f"'{node}' is not a node of the graph")
        visits = [node]
        for _ in range(steps):
            nbrs = sorted(self.graph.neighbors(node))
            if not nbrs:
                self.restarts += 1
                node = self.rng.choice(self.nodes)
            else:
                node = self.rng.choice(nbrs)
            visits.append(node)
        return visits

    def by_walk(self, steps: int, start: str | None = None) -> Graph:
        return self.induced(set(self.walk(steps, start)))

    def visit_shares(self, steps: int, start: str | None = None) -> dict[str, float]:
        visits = Counter(self.walk(steps, start))
        total = sum(visits.values())
        return {n: visits[n] / total for n in self.nodes}

    def corrected_shares(self, steps: int, start: str | None = None) -> dict[str, float]:
        # weighting each visit by one over degree undoes the walk's pull toward hubs
        visits = Counter(self.walk(steps, start))
        degree = {n: self.graph.degree(n) for n in self.nodes}
        weights = {n: visits[n] / degree[n] for n in self.nodes if degree[n]}
        total = sum(weights.values())
        return {n: w / total for n, w in weights.items()} if total else {}

    def degree_bias(self, sample: Graph) -> float:
        if sample.node_count() == 0 or not self.nodes:
            return 0.0
        inside = sum(self.graph.degree(n) for n in sample.nodes()) / sample.node_count()
        overall = sum(self.graph.degree(n) for n in self.nodes) / len(self.nodes)
        return inside / overall if overall else 0.0

    def note(self, sample: Graph, method: str) -> str:
        return (
            f"{method} sample of {sample.node_count()} node(s) and {sample.edge_count()} "
            f"edge(s); mean original degree {self.degree_bias(sample):.2f}x the graph's"
        )
