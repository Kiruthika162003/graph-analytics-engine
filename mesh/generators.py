"""Random graph generators: three recipes, each with a signature you can measure.

Analyzing an algorithm needs graphs to run it on, and the three classic
random models each produce a recognizably different kind. Erdos-Renyi
places every possible edge independently with probability p: degrees
cluster tightly around their mean of p times n minus one, there is
almost no clustering beyond p itself, and above a sharp threshold near
one over n a giant component appears. Barabasi-Albert grows the graph by
preferential attachment: each new node brings m edges and attaches to
existing nodes with probability proportional to their current degree, so
the rich get richer and the degree distribution grows a heavy tail, a few
hubs of very high degree where Erdos-Renyi would have none, the pattern
of citation networks and the web. Watts-Strogatz starts from a ring where
every node links to its k nearest neighbors, which has high clustering
and long paths, then rewires each edge with probability beta to a random
target; a small beta keeps the clustering while the few long-range
shortcuts collapse the average path length, the small-world combination
that neither of the other models gives. The engine builds all three from
a seed so each is reproducible, and it exposes the measurements that
tell them apart: the maximum degree against the mean, which is modest for
Erdos-Renyi and large for Barabasi-Albert, and the clustering
coefficient, which is near p for Erdos-Renyi and far above it for a
lightly rewired Watts-Strogatz. It refuses parameters that make a model
meaningless, a probability outside zero to one, more edges per new node
than nodes to attach to, an odd neighbor count on the ring. It reports
each graph's max-degree ratio and clustering, because those two numbers
are the model's fingerprint and a graph whose fingerprint does not match
its label came from a different process than the caller believed.
"""

from __future__ import annotations

import random

from mesh.errors import Invalid
from mesh.graph import Graph
from mesh.triangles import Triangles


def erdos_renyi(n: int, p: float, seed: int = 0) -> Graph:
    if n < 1:
        raise Invalid("a graph needs at least one node")
    if not 0.0 <= p <= 1.0:
        raise Invalid("the edge probability must lie in [0, 1]")
    rng = random.Random(seed)
    g = Graph()
    nodes = [str(i) for i in range(n)]
    for node in nodes:
        g.add_node(node)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                g.add_edge(nodes[i], nodes[j])
    return g


def barabasi_albert(n: int, m: int, seed: int = 0) -> Graph:
    if m < 1 or m >= n:
        raise Invalid("each new node needs between 1 and n-1 attachments")
    rng = random.Random(seed)
    g = Graph()
    # seed with a complete core of m+1 nodes so every early node has degree
    core = [str(i) for i in range(m + 1)]
    for node in core:
        g.add_node(node)
    for i in range(len(core)):
        for j in range(i + 1, len(core)):
            g.add_edge(core[i], core[j])
    # a target list with each node repeated by its degree makes the
    # preferential draw a uniform pick
    targets: list[str] = []
    for node in core:
        targets.extend([node] * g.degree(node))
    for i in range(m + 1, n):
        node = str(i)
        g.add_node(node)
        chosen: set[str] = set()
        while len(chosen) < m:
            chosen.add(rng.choice(targets))
        for t in chosen:
            g.add_edge(node, t)
            targets.extend([node, t])
    return g


def watts_strogatz(n: int, k: int, beta: float, seed: int = 0) -> Graph:
    if k % 2 != 0 or k < 2 or k >= n:
        raise Invalid("the ring degree k must be even, at least 2, and below n")
    if not 0.0 <= beta <= 1.0:
        raise Invalid("the rewiring probability must lie in [0, 1]")
    rng = random.Random(seed)
    g = Graph()
    nodes = [str(i) for i in range(n)]
    for node in nodes:
        g.add_node(node)
    for i in range(n):
        for step in range(1, k // 2 + 1):
            g.add_edge(nodes[i], nodes[(i + step) % n])
    # rewire each ring edge's far end with probability beta
    rewired = Graph()
    for node in nodes:
        rewired.add_node(node)
    for u, v, _w in g.edges():
        target = v
        if rng.random() < beta:
            candidates = [x for x in nodes if x != u and not rewired.has_edge(u, x)]
            if candidates:
                target = rng.choice(candidates)
        if not rewired.has_edge(u, target) and u != target:
            rewired.add_edge(u, target)
    return rewired


def fingerprint(g: Graph) -> dict[str, float]:
    degrees = [g.degree(n) for n in g.nodes()]
    mean = sum(degrees) / len(degrees)
    return {
        "max_over_mean": max(degrees) / mean if mean else 0.0,
        "clustering": Triangles(g).average_clustering(),
    }


def note(g: Graph, label: str) -> str:
    f = fingerprint(g)
    return (
        f"{label}: max degree at {f['max_over_mean']:.2f}x the mean, clustering "
        f"{f['clustering']:.3f}; a fingerprint that does not match its label came "
        "from a different process than believed"
    )
