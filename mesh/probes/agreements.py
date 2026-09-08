"""Agreement probes: two independent algorithms forced to land on one number.

Each probe here builds a graph designed to be awkward and runs two methods
that must agree if both are right. Agreement between unrelated
implementations is a stronger check than either against a hand-typed
expectation, because a mistaken expectation and a mistaken implementation
can agree with each other while two algorithms of different shape cannot
share a bug by accident.
"""

from __future__ import annotations

import random
from itertools import pairwise

from mesh.bellmanford import BellmanFord
from mesh.dijkstra import Dijkstra
from mesh.edmondskarp import EdmondsKarp
from mesh.graph import Graph
from mesh.kosarajuscc import KosarajuSCC
from mesh.kruskal import Kruskal
from mesh.prim import Prim
from mesh.probes.probe import Probe
from mesh.probes.registry import register
from mesh.tarjanscc import TarjanSCC


def _random_digraph(seed: int, n: int, p: float, low: int, high: int) -> Graph:
    rng = random.Random(seed)
    g = Graph(directed=True)
    nodes = [str(i) for i in range(n)]
    for node in nodes:
        g.add_node(node)
    for u in nodes:
        for v in nodes:
            if u != v and rng.random() < p:
                g.add_edge(u, v, rng.randint(low, high))
    return g


@register
def dijkstra_meets_bellman_ford() -> Probe:
    # on non-negative weights the heap and the relaxation passes must agree
    g = _random_digraph(seed=1, n=12, p=0.3, low=1, high=9)
    dij = Dijkstra(g, "0")
    bf = BellmanFord(g, "0")
    compared = 0
    disagreements = 0
    for node in g.nodes():
        reachable = bf.distance[node] != float("inf")
        if reachable != (node in dij.distance):
            disagreements += 1
        elif reachable:
            compared += 1
            if dij.distance_to(node) != bf.distance_to(node):
                disagreements += 1
    return Probe(
        prober="pathsagree",
        guarantee="Dijkstra and Bellman-Ford give identical distances with no negative edge",
        holds=disagreements == 0,
        readings={"compared": compared, "disagreements": disagreements},
    )


@register
def flow_equals_cut() -> Probe:
    # max-flow min-cut: the flow must equal the cheapest source-side subset cut
    rng = random.Random(2)
    g = Graph(directed=True)
    nodes = ["s", "a", "b", "c", "d", "t"]
    for n in nodes:
        g.add_node(n)
    for u in nodes:
        for v in nodes:
            if v not in (u, "s") and u != "t" and rng.random() < 0.5:
                g.add_edge(u, v, rng.randint(1, 10))
    flow = EdmondsKarp(g, "s", "t").value
    interior = ["a", "b", "c", "d"]
    best_cut = float("inf")
    for mask in range(1 << len(interior)):
        side = {"s"} | {interior[i] for i in range(4) if mask >> i & 1}
        cut = sum(w for u, v, w in g.edges() if u in side and v not in side)
        best_cut = min(best_cut, cut)
    return Probe(
        prober="flowcut",
        guarantee="the maximum flow equals the minimum cut, checked over every cut",
        holds=flow == best_cut,
        readings={"flow": flow, "min_cut": best_cut, "cuts_checked": 1 << 4},
    )


@register
def spanning_trees_agree() -> Probe:
    # two greedy routes, edge-by-edge and grown-from-a-node, reach one minimum
    rng = random.Random(3)
    g = Graph()
    nodes = [str(i) for i in range(10)]
    for n in nodes:
        g.add_node(n)
    shuffled = nodes[:]
    rng.shuffle(shuffled)
    for a, b in pairwise(shuffled):
        g.add_edge(a, b, rng.randint(1, 20))
    for _ in range(15):
        a, b = rng.sample(nodes, 2)
        if not g.has_edge(a, b):
            g.add_edge(a, b, rng.randint(1, 20))
    k = Kruskal(g).total_weight()
    p = Prim(g).total_weight()
    return Probe(
        prober="treesagree",
        guarantee="Kruskal and Prim find spanning trees of the same minimum weight",
        holds=k == p,
        readings={"kruskal": k, "prim": p, "edges": g.edge_count()},
    )


@register
def components_agree() -> Probe:
    # one-pass low-link and two-pass reverse-DFS must partition identically
    g = _random_digraph(seed=4, n=14, p=0.2, low=1, high=1)
    t = TarjanSCC(g)
    k = KosarajuSCC(g)
    same = t.count() == k.count() and all(
        t.component_of(n) == k.component_of(n) for n in g.nodes()
    )
    return Probe(
        prober="sccagree",
        guarantee="Tarjan and Kosaraju produce the identical strongly connected partition",
        holds=same,
        readings={"components": t.count(), "nodes": g.node_count()},
    )
