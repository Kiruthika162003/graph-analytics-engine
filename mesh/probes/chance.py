"""Chance probes: seeded randomness held to the exact facts that do not depend on the seed.

Every shuffle keeps its degrees, percolation reads its two ends
exactly, an epidemic with certain transmission sweeps a connected
graph entirely, a random walk visits in proportion to degree, and a
ring of cliques clusters beyond its null.
"""

from __future__ import annotations

import random
from itertools import combinations

from mesh.edgeswap import EdgeSwap
from mesh.epidemic import Epidemic
from mesh.factories import cycle, path
from mesh.graph import Graph
from mesh.nullmodel import NullModel
from mesh.percolation import Percolation
from mesh.probes.probe import Probe
from mesh.probes.registry import register
from mesh.sampling import Sampler


def _random_connected(seed: int, n: int, p: float) -> Graph:
    rng = random.Random(seed)
    g = path(n)
    for a, b in combinations(g.nodes(), 2):
        if not g.has_edge(a, b) and rng.random() < p:
            g.add_edge(a, b)
    return g


def _ring_of_cliques(cliques: int, size: int) -> Graph:
    g = Graph()
    for c in range(cliques):
        names = [f"c{c}n{i}" for i in range(size)]
        for n in names:
            g.add_node(n)
        for a, b in combinations(names, 2):
            g.add_edge(a, b)
    for c in range(cliques):
        g.add_edge(f"c{c}n0", f"c{(c + 1) % cliques}n1")
    return g


@register
def every_shuffle_keeps_its_degrees() -> Probe:
    # five hundred swaps on a random graph must leave every degree in place
    g = _random_connected(seed=107, n=16, p=0.2)
    swapper = EdgeSwap(g, seed=1)
    out = swapper.shuffle(500)
    return Probe(
        prober="edgeswap",
        guarantee="a degree-preserving shuffle leaves every degree and the edge count in place",
        holds=swapper.degrees_preserved(out) and out.edge_count() == g.edge_count(),
        readings={"accepted": swapper.accepted, "edges": g.edge_count()},
    )


@register
def percolation_reads_its_ends_exactly() -> Probe:
    # keep nothing and the giant piece is one node; keep all and it is the graph
    g = _random_connected(seed=109, n=12, p=0.25)
    pc = Percolation(g, seed=2)
    low = pc.giant_fraction(0.0, trials=3)
    high = pc.giant_fraction(1.0, trials=3)
    return Probe(
        prober="percolation",
        guarantee="the giant piece is one node at retention zero and everything at one",
        holds=abs(low - 1 / 12) < 1e-12 and abs(high - 1.0) < 1e-12,
        readings={"at_zero": round(low, 4), "at_one": round(high, 4)},
    )


@register
def certain_transmission_sweeps_a_connected_graph() -> Probe:
    # with beta one every reachable node is infected, whatever the seed does
    g = _random_connected(seed=113, n=12, p=0.25)
    ep = Epidemic(g, beta=1.0, gamma=0.5, seed=3)
    sizes = [ep.run()[0] for _ in range(5)]
    return Probe(
        prober="epidemic",
        guarantee="certain transmission on a connected graph infects every node in every trial",
        holds=all(s == g.node_count() for s in sizes),
        readings={"sizes": sizes, "nodes": g.node_count()},
    )


@register
def walk_visits_track_degree_share() -> Probe:
    # on a hub with a ring, the hub's visit share approaches its degree share
    g = cycle(10)
    g.add_node("hub")
    for n in cycle(10).nodes():
        g.add_edge("hub", n)
    shares = Sampler(g, seed=4).visit_shares(20000, start="hub")
    expected = g.degree("hub") / (2 * g.edge_count())
    return Probe(
        prober="sampling",
        guarantee="a long random walk visits a node in proportion to its degree share",
        holds=abs(shares["hub"] - expected) < 0.03,
        readings={"observed": round(shares["hub"], 4), "expected": round(expected, 4)},
    )


@register
def ring_of_cliques_clusters_beyond_its_null() -> Probe:
    # the observed transitivity must sit far above the shuffled mean
    nm = NullModel(_ring_of_cliques(6, 4), samples=15, seed=5)
    t = nm.transitivity()
    return Probe(
        prober="nullmodel",
        guarantee="a ring of cliques clusters beyond what its degree sequence alone gives",
        holds=t["z"] > 2 and t["p"] == 0.0 and nm.degrees_preserved,
        readings={"observed": round(t["observed"], 3), "null_mean": round(t["mean"], 3)},
    )
