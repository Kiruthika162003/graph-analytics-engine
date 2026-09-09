"""Walk probes: integer walk counts and brokerage held to their exact small cases.

Length-two walks are common neighbors, traces match the spectrum, and
a star hub's constraint is one over its leaves.
"""

from __future__ import annotations

import random
from itertools import combinations

from mesh.factories import star
from mesh.graph import Graph
from mesh.neighborhood import Neighborhood
from mesh.probes.probe import Probe
from mesh.probes.registry import register
from mesh.walkcount import WalkCount


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
def two_step_walks_are_common_neighbors() -> Probe:
    # the square of the adjacency counts common neighbors off the diagonal
    g = _random_undirected(seed=173, n=9, p=0.45)
    wc = WalkCount(g)
    pairs = list(combinations(g.nodes(), 2))
    matched = sum(1 for a, b in pairs if wc.walks(a, b, 2) == wc.common_neighbors(a, b))
    return Probe(
        prober="walkcount",
        guarantee="walks of length two between distinct nodes count their common neighbors",
        holds=matched == len(pairs),
        readings={"matched": matched, "pairs": len(pairs)},
    )


@register
def walk_traces_match_the_spectrum() -> Probe:
    # integer traces and eigenvalue power sums must agree at lengths two to five
    g = _random_undirected(seed=179, n=8, p=0.5)
    wc = WalkCount(g)
    agreed = [k for k in (2, 3, 4, 5) if wc.matches_spectrum(k)]
    return Probe(
        prober="walkcount",
        guarantee="the trace of each adjacency power equals the eigenvalue power sum",
        holds=agreed == [2, 3, 4, 5],
        readings={"agreed": agreed, "trace3": wc.trace(3)},
    )


@register
def star_hub_constraint_is_one_over_leaves() -> Probe:
    # contacts who are strangers to one another leave the hub free to broker
    nb = Neighborhood(star(8))
    constraint = nb.constraint("0")
    return Probe(
        prober="neighborhood",
        guarantee="a star hub's constraint is one over its leaf count with full effective size",
        holds=abs(constraint - 1 / 8) < 1e-12 and nb.effective_size("0") == 8.0,
        readings={"constraint": round(constraint, 6), "effective": nb.effective_size("0")},
    )
