"""Invariant probes: theorems the engine's own numbers must satisfy.

Where the agreement probes force two algorithms onto one number, these
check a single result against a law it cannot break: a nullity that
must equal a component count, a runner-up that must not beat the winner,
a stretch that must stay under its bound, a chromatic number bracketed
by a clique and a greedy, and Whitney's inequality chain.
"""

from __future__ import annotations

import random
from itertools import combinations, pairwise

from mesh.chromatic import ChromaticNumber
from mesh.connectedcomponents import ConnectedComponents
from mesh.connectivitynumbers import ConnectivityNumbers
from mesh.graph import Graph
from mesh.laplacian import Laplacian
from mesh.probes.probe import Probe
from mesh.probes.registry import register
from mesh.secondbestmst import SecondBestMST
from mesh.spanner import GreedySpanner


def _random_undirected(seed: int, n: int, p: float, low: int, high: int) -> Graph:
    rng = random.Random(seed)
    g = Graph()
    nodes = [str(i) for i in range(n)]
    for node in nodes:
        g.add_node(node)
    for a, b in combinations(nodes, 2):
        if rng.random() < p:
            g.add_edge(a, b, rng.randint(low, high))
    return g


@register
def nullity_counts_components() -> Probe:
    # the Laplacian's null space has one dimension per connected component
    g = _random_undirected(seed=11, n=12, p=0.15, low=1, high=5)
    nullity = Laplacian(g).nullity()
    components = ConnectedComponents(g).count()
    return Probe(
        prober="nullity",
        guarantee="the Laplacian nullity equals the connected component count",
        holds=nullity == components,
        readings={"nullity": nullity, "components": components},
    )


@register
def runner_up_never_beats_winner() -> Probe:
    # a second-best spanning tree costs at least as much as the best
    rng = random.Random(12)
    g = Graph()
    nodes = [str(i) for i in range(9)]
    for n in nodes:
        g.add_node(n)
    shuffled = nodes[:]
    rng.shuffle(shuffled)
    for a, b in pairwise(shuffled):
        g.add_edge(a, b, rng.randint(1, 20))
    for _ in range(10):
        a, b = rng.sample(nodes, 2)
        if not g.has_edge(a, b):
            g.add_edge(a, b, rng.randint(1, 20))
    sb = SecondBestMST(g)
    return Probe(
        prober="runnerup",
        guarantee="the second-best spanning tree never costs less than the best",
        holds=sb.second >= sb.best,
        readings={"best": sb.best, "second": sb.second},
    )


@register
def spanner_keeps_its_promise() -> Probe:
    # every pair's distance in the spanner is within the stretch factor
    g = _random_undirected(seed=13, n=10, p=0.6, low=1, high=15)
    sp = GreedySpanner(g, stretch=2.0)
    observed = sp.observed_stretch()
    return Probe(
        prober="stretch",
        guarantee="a greedy 2-spanner stretches no distance by more than two",
        holds=observed <= 2.0 + 1e-9,
        readings={"observed": round(observed, 3), "kept": sp.kept(), "of": g.edge_count()},
    )


@register
def chromatic_sits_between_its_bounds() -> Probe:
    # a clique needs its size in colors; greedy is always enough
    g = _random_undirected(seed=14, n=10, p=0.45, low=1, high=1)
    cn = ChromaticNumber(g)
    return Probe(
        prober="chromatic",
        guarantee="the chromatic number lies between the clique number and the greedy count",
        holds=cn.lower <= cn.value <= cn.upper,
        readings={"clique": cn.lower, "chromatic": cn.value, "greedy": cn.upper},
    )


@register
def whitney_chain_holds() -> Probe:
    # vertex connectivity <= edge connectivity <= minimum degree
    g = _random_undirected(seed=15, n=8, p=0.5, low=1, high=1)
    cn = ConnectivityNumbers(g)
    return Probe(
        prober="whitney",
        guarantee="vertex connectivity is at most edge connectivity is at most min degree",
        holds=cn.whitney_holds(),
        readings={
            "vertex": cn.vertex_connectivity,
            "edge": cn.edge_connectivity,
            "min_degree": cn.min_degree,
        },
    )
