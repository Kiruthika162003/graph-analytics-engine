"""Hardness probes: exact answers to hard problems held beside their cheap bounds.

Pathwidth against every ordering, feedback sets against greedy, the
double-tree tour within twice the exact one, burning against the root-n
conjecture on a tree, and Freeman centralization pinned at one on a star.
"""

from __future__ import annotations

import random
from itertools import combinations, permutations
from math import hypot

from mesh.burning import Burning
from mesh.centralization import Centralization
from mesh.factories import star
from mesh.feedbackvertexset import FeedbackVertexSet
from mesh.graph import Graph
from mesh.pathwidth import Pathwidth
from mesh.probes.probe import Probe
from mesh.probes.registry import register
from mesh.prufer import Prufer
from mesh.tsp import TravellingSalesman


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
def pathwidth_dynamic_matches_every_ordering() -> Probe:
    # the subset dynamic must equal the minimum over all node orderings
    g = _random_undirected(seed=89, n=7, p=0.4)
    pw = Pathwidth(g)
    brute = min(pw.separation(list(p)) for p in permutations(g.nodes()))
    return Probe(
        prober="pathwidth",
        guarantee="the pathwidth dynamic equals the best vertex separation over every ordering",
        holds=pw.exact() == brute,
        readings={"dynamic": pw.exact(), "orderings": brute},
    )


@register
def greedy_feedback_set_never_beats_exact() -> Probe:
    # both must break every cycle and the greedy must not be smaller
    g = _random_undirected(seed=97, n=9, p=0.3)
    fvs = FeedbackVertexSet(g)
    exact, greedy = fvs.exact(), fvs.greedy()
    return Probe(
        prober="feedbackvertexset",
        guarantee="greedy and exact feedback sets both break every cycle, greedy never smaller",
        holds=fvs.breaks_every_cycle(exact)
        and fvs.breaks_every_cycle(greedy)
        and len(greedy) >= len(exact),
        readings={"exact": len(exact), "greedy": len(greedy)},
    )


@register
def double_tree_tour_within_twice_optimum() -> Probe:
    # with the triangle inequality the shortcut walk is at most twice the exact tour
    rng = random.Random(101)
    pts = {f"p{i}": (rng.uniform(0, 10), rng.uniform(0, 10)) for i in range(9)}
    g = Graph()
    for name in pts:
        g.add_node(name)
    for a, b in combinations(list(pts), 2):
        g.add_edge(a, b, hypot(pts[a][0] - pts[b][0], pts[a][1] - pts[b][1]))
    ts = TravellingSalesman(g)
    ratio = ts.ratio()
    return Probe(
        prober="tsp",
        guarantee="the double-tree tour is within twice the exact tour on points in the plane",
        holds=ts.is_tour(ts.double_tree()) and 1.0 - 1e-9 <= ratio <= 2.0 + 1e-9,
        readings={"ratio": round(ratio, 4)},
    )


@register
def burning_respects_root_n_on_a_tree() -> Probe:
    # the burning number of a tree is at most the ceiling of root n
    rng = random.Random(103)
    tree = Prufer.decode([rng.randrange(11) for _ in range(9)])
    b = Burning(tree)
    rounds = len(b.exact())
    return Probe(
        prober="burning",
        guarantee="a random tree burns within the ceiling of the square root of its size",
        holds=rounds <= b.conjecture_bound() and b.burns(b.greedy()),
        readings={"rounds": rounds, "bound": b.conjecture_bound()},
    )


@register
def star_centralization_is_one() -> Probe:
    # the star is the ceiling of every reading, so it must score exactly one
    c = Centralization(star(7))
    readings = {"degree": c.degree(), "betweenness": c.betweenness(), "harmonic": c.harmonic()}
    return Probe(
        prober="centralization",
        guarantee="a star scores one on degree, betweenness, and harmonic centralization",
        holds=all(abs(v - 1.0) < 1e-9 for v in readings.values()),
        readings={k: round(v, 6) for k, v in readings.items()},
    )
