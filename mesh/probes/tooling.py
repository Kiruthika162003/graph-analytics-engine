"""Tooling probes: the supporting modules held to the algebra a reader relies on.

Merge counts obey inclusion-exclusion, a relabeling round trip returns
an equal graph, the cache computes once per version, a layered layout
points every arc downward, and a path's steps sum to its distance.
"""

from __future__ import annotations

import random
from itertools import combinations

from mesh.graph import Graph
from mesh.graphcache import ReadingCache
from mesh.graphio import same_graph
from mesh.graphmerge import GraphMerge
from mesh.layout import Layout
from mesh.pathexplain import PathExplanation
from mesh.probes.probe import Probe
from mesh.probes.registry import register
from mesh.relabel import Relabel


def _random_undirected(seed: int, n: int, p: float) -> Graph:
    rng = random.Random(seed)
    g = Graph()
    nodes = [str(i) for i in range(n)]
    for node in nodes:
        g.add_node(node)
    for a, b in combinations(nodes, 2):
        if rng.random() < p:
            g.add_edge(a, b, float(rng.randint(1, 4)))
    return g


@register
def merge_counts_obey_inclusion_exclusion() -> Probe:
    # union equals the two counts minus the intersection, on a random pair
    a = _random_undirected(seed=127, n=8, p=0.4)
    b = _random_undirected(seed=131, n=8, p=0.4)
    gm = GraphMerge(a, b)
    return Probe(
        prober="graphmerge",
        guarantee="the union's edge count is the two counts minus the intersection's",
        holds=gm.counts_agree(),
        readings={"union": gm.union().edge_count(), "shared": gm.intersection().edge_count()},
    )


@register
def relabel_round_trip_returns_an_equal_graph() -> Probe:
    # prefixing then applying the inverse mapping must give the original back
    g = _random_undirected(seed=137, n=7, p=0.5)
    out, inverse = Relabel(g).with_prefix("node:")
    back, _inv = Relabel(out).by_mapping(inverse)
    return Probe(
        prober="relabel",
        guarantee="relabeling by a mapping and then by its inverse gives back an equal graph",
        holds=same_graph(back, g) and out.edge_count() == g.edge_count(),
        readings={"nodes": g.node_count(), "edges": g.edge_count()},
    )


@register
def cache_computes_once_per_version() -> Probe:
    # two asks on one graph cost one computation; an edit costs one more
    calls = {"n": 0}

    def reading(graph: Graph) -> int:
        calls["n"] += 1
        return graph.edge_count()

    cache = ReadingCache()
    cache.register("edges", reading)
    g = _random_undirected(seed=139, n=6, p=0.5)
    cache.get("edges", g)
    cache.get("edges", g)
    g.add_node("extra")
    cache.get("edges", g)
    return Probe(
        prober="graphcache",
        guarantee="a cached reading is computed once per graph version and again after an edit",
        holds=calls["n"] == 2 and cache.hits == 1,
        readings={"computed": calls["n"], "hits": cache.hits},
    )


@register
def layered_layout_points_every_arc_downward() -> Probe:
    # in a layered drawing of a DAG, every arc's head sits strictly below its tail
    g = Graph(directed=True)
    for n in ("a", "b", "c", "d", "e", "f"):
        g.add_node(n)
    for u, v in [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d"), ("d", "e"), ("c", "f")]:
        g.add_edge(u, v)
    pos = Layout(g).layered()
    downward = sum(1 for u, v, _w in g.edges() if pos[v][1] > pos[u][1])
    return Probe(
        prober="layout",
        guarantee="a layered layout of a DAG puts every arc's head strictly below its tail",
        holds=downward == g.edge_count() and Layout.inside_square(pos),
        readings={"downward": downward, "arcs": g.edge_count()},
    )


@register
def path_steps_sum_to_the_distance() -> Probe:
    # the explanation's steps must add up to what Dijkstra reported
    g = _random_undirected(seed=149, n=9, p=0.35)
    for a, b in zip(g.nodes(), g.nodes()[1:], strict=False):
        if not g.has_edge(a, b):
            g.add_edge(a, b, 2.0)
    pe = PathExplanation(g, "0", "8")
    runner = pe.runner_up()
    return Probe(
        prober="pathexplain",
        guarantee="a route's steps sum to its distance and the runner-up is never shorter",
        holds=pe.steps_sum_to_distance() and (runner is None or runner[1] >= pe.distance),
        readings={"distance": pe.distance, "steps": len(pe.steps())},
    )
