"""Random walk: wander the graph, and the time spent per node tends to its degree.

A random walk starts at a node and at every step moves to a uniformly
chosen neighbor. Where it spends its time is not uniform: on a connected
undirected graph the long-run fraction of steps at a node tends to that
node's degree divided by twice the edge count, so a node with three times
the degree of another is visited three times as often. That stationary
distribution is exact and it is the same fact that makes PageRank without
damping proportional to degree on an undirected graph. The walk is worth
having as a simulation because it is the ground truth the analytic
distribution is claiming to describe, and the two are compared here rather
than one taken on faith: run a long walk, count visits, and measure how
far the empirical frequencies sit from the degree-proportional prediction.
The gap shrinks with the length of the walk at a rate of roughly one over
the square root of the steps, so a short walk can disagree with theory
noticeably, and a first guess that a few thousand steps is plenty gets
refuted on a graph with a poorly connected corner the walk rarely enters.
A bipartite graph is the one exception to convergence: the walk alternates
sides every step, so the fraction of time on a side never settles, and the
engine flags that rather than reporting a distribution that depends on
whether the step count was odd or even. The walk takes a seed for
reproducibility, runs a given number of steps, returns the visit counts
and empirical frequencies, the theoretical stationary distribution, and
the total variation distance between them, and reports that distance
against one over root steps, because a distance well above that scale is
a walk that has not mixed yet, most often from a bottleneck it keeps
failing to cross.
"""

from __future__ import annotations

import math
import random

from mesh.bipartite import Bipartite
from mesh.connectedcomponents import ConnectedComponents
from mesh.errors import Invalid, Missing
from mesh.graph import Graph


class RandomWalk:
    def __init__(self, graph: Graph, start: str, steps: int, seed: int = 0) -> None:
        if graph.directed:
            raise Invalid("the degree-proportional stationary law is for undirected graphs")
        if not graph.has_node(start):
            raise Missing(f"start '{start}' is not in the graph")
        if steps < 1:
            raise Invalid("a walk needs at least one step")
        if not ConnectedComponents(graph).is_connected():
            raise Invalid("the walk cannot reach a disconnected component")
        self.graph = graph
        self.steps = steps
        self.alternates = Bipartite(graph).is_bipartite
        self.visits: dict[str, int] = dict.fromkeys(graph.nodes(), 0)
        rng = random.Random(seed)
        node = start
        for _ in range(steps):
            nbrs = list(graph.neighbors(node))
            node = rng.choice(nbrs)
            self.visits[node] += 1

    def empirical(self) -> dict[str, float]:
        return {n: c / self.steps for n, c in self.visits.items()}

    def stationary(self) -> dict[str, float]:
        # each node's share is its degree over twice the edge count
        two_m = 2 * self.graph.edge_count()
        return {n: self.graph.degree(n) / two_m for n in self.graph.nodes()}

    def total_variation(self) -> float:
        emp = self.empirical()
        theory = self.stationary()
        return 0.5 * sum(abs(emp[n] - theory[n]) for n in self.graph.nodes())

    def noise_scale(self) -> float:
        return 1.0 / math.sqrt(self.steps)

    def note(self) -> str:
        if self.alternates:
            return "bipartite: the walk alternates sides and its side share never settles"
        tv = self.total_variation()
        scale = self.noise_scale()
        verdict = "mixed" if tv < 3 * scale else "not yet mixed, likely a bottleneck"
        return (
            f"total variation {tv:.4f} against noise scale {scale:.4f} after "
            f"{self.steps} step(s): {verdict}"
        )
