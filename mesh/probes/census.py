"""Census probes: four-node shapes and the benchmark frame held to their counts.

Quad counts sum to n choose 4 and the triangles they imply match a
direct count, a complete graph is all complete quads, and a counting
reading grows exactly with the edge count.
"""

from __future__ import annotations

import random
from itertools import combinations
from math import comb

from mesh.benchmark import Benchmark
from mesh.factories import complete, cycle
from mesh.graph import Graph
from mesh.probes.probe import Probe
from mesh.probes.registry import register
from mesh.quadcensus import QuadCensus


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
def quads_sum_and_triangles_agree() -> Probe:
    # every four-subset lands somewhere, and the paws, diamonds, and K4s imply the triangles
    g = _random_undirected(seed=151, n=10, p=0.45)
    qc = QuadCensus(g)
    return Probe(
        prober="quadcensus",
        guarantee="quad counts sum to n choose 4 and imply the direct triangle count",
        holds=qc.sums_to_choose_four() and qc.triangle_identity_holds(),
        readings={"total": qc.total(), "triangles": qc.triangles_by_search()},
    )


@register
def complete_graph_is_all_complete_quads() -> Probe:
    # K7 has 7 choose 4 complete quads and nothing else
    qc = QuadCensus(complete(7))
    return Probe(
        prober="quadcensus",
        guarantee="a complete graph's quads are all complete and number n choose 4",
        holds=qc.counts["complete"] == comb(7, 4) and qc.disconnected == 0,
        readings={"complete": qc.counts["complete"], "expected": comb(7, 4)},
    )


@register
def edge_visits_grow_linearly() -> Probe:
    # a reading that visits each edge once must grow like the edge count
    def visits(g: Graph) -> int:
        return sum(1 for _e in g.edges())

    bench = Benchmark(cycle, [10, 100]).add("visits", visits, visits)
    bench.run()
    exponent = bench.growth("visits")
    return Probe(
        prober="benchmark",
        guarantee="a reading that visits every edge once is read as linear growth",
        holds=exponent is not None and abs(exponent - 1.0) < 1e-9,
        readings={"exponent": round(exponent, 6) if exponent is not None else None},
    )
