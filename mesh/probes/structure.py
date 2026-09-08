"""Structure probes: matchings, separators, codes, and trails held to their theorems.

Konig's theorem, Menger's theorem, the Prufer bijection, Berge's
theorem through the blossom matcher, and the balance rule for directed
Eulerian trails, each checked on a graph built to make the theorem do
some work.
"""

from __future__ import annotations

import random
from itertools import combinations, pairwise

from mesh.blossom import Blossom
from mesh.eulerdirected import DirectedEulerian
from mesh.graph import Graph
from mesh.menger import Menger
from mesh.probes.probe import Probe
from mesh.probes.registry import register
from mesh.prufer import Prufer
from mesh.vertexcover import VertexCover
from mesh.vertexseparator import VertexSeparator


def _random_undirected(seed: int, n: int, p: float) -> Graph:
    rng = random.Random(seed)
    g = Graph()
    nodes = [str(i) for i in range(n)]
    for node in nodes:
        g.add_node(node)
    for a, b in combinations(nodes, 2):
        if rng.random() < p:
            g.add_edge(a, b)
    return g


@register
def konig_cover_equals_matching() -> Probe:
    # on a bipartite graph the minimum vertex cover has the matching's size
    rng = random.Random(21)
    g = Graph()
    left = [f"l{i}" for i in range(6)]
    right = [f"r{i}" for i in range(6)]
    for n in left + right:
        g.add_node(n)
    for u in left:
        for v in right:
            if rng.random() < 0.4:
                g.add_edge(u, v)
    vc = VertexCover(g)
    return Probe(
        prober="konig",
        guarantee="on a bipartite graph the minimum vertex cover equals the maximum matching",
        holds=vc.exact and len(vc.cover) == vc.lower_bound and vc.covers_every_edge(),
        readings={"cover": len(vc.cover), "matching": vc.lower_bound},
    )


@register
def menger_separator_matches_count() -> Probe:
    # the separator's size must equal the node-disjoint route count; seeds 22 and 23
    # both joined the ends directly, so the seed advances until the ends are apart
    seed = 22
    g = _random_undirected(seed=seed, n=9, p=0.35)
    while g.has_edge("0", "8"):
        seed += 1
        g = _random_undirected(seed=seed, n=9, p=0.35)
    vs = VertexSeparator(g, "0", "8")
    routes = Menger(g, "0", "8").node_disjoint
    return Probe(
        prober="menger",
        guarantee="the minimum vertex separator is as large as the count of disjoint routes",
        holds=vs.separates() and len(vs.separator) == routes,
        readings={"separator": len(vs.separator), "routes": routes},
    )


@register
def prufer_round_trip() -> Probe:
    # decode then encode must return the very same sequence
    rng = random.Random(24)
    code = [rng.randrange(15) for _ in range(13)]
    tree = Prufer.decode(code)
    back = Prufer.encode(tree)
    return Probe(
        prober="prufer",
        guarantee="a Prufer sequence decodes to a tree that encodes back to the same sequence",
        holds=back == code and tree.edge_count() == 14,
        readings={"length": len(code), "edges": tree.edge_count()},
    )


@register
def blossom_matches_enumeration() -> Probe:
    # Berge through Edmonds: the blossom size equals the brute-force maximum
    g = _random_undirected(seed=25, n=8, p=0.4)
    edges = [(u, v) for u, v, _w in g.edges()]
    best = 0
    for k in range(len(edges), 0, -1):
        for subset in combinations(edges, k):
            ends = [n for e in subset for n in e]
            if len(set(ends)) == 2 * k:
                best = k
                break
        if best:
            break
    size = Blossom(g).size()
    return Probe(
        prober="blossom",
        guarantee="the blossom matcher finds a matching as large as exhaustive search",
        holds=size == best,
        readings={"blossom": size, "exhaustive": best, "edges": len(edges)},
    )


@register
def directed_trail_covers_every_arrow() -> Probe:
    # a balanced strongly connected digraph has a circuit using each arrow once
    g = Graph(directed=True)
    for n in "abcde":
        g.add_node(n)
    for u, v in [("a", "b"), ("b", "c"), ("c", "a"), ("c", "d"), ("d", "e"), ("e", "c")]:
        g.add_edge(u, v)
    trail = DirectedEulerian(g).trail
    steps = list(pairwise(trail))
    arrows = sorted((u, v) for u, v, _w in g.edges())
    return Probe(
        prober="eulerdirected",
        guarantee="a balanced strongly connected digraph has a circuit walking each arrow once",
        holds=sorted(steps) == arrows and trail[0] == trail[-1],
        readings={"arrows": g.edge_count(), "steps": len(steps)},
    )
