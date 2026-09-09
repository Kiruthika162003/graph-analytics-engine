"""Counting probes: polynomials and counts held to their closed forms and to each other.

The chromatic polynomial of a cycle, spanning trees by two methods
that share no code, perfect matchings by recursion and by Ryser, the
independence polynomial against enumeration, and the triad census
summing to n choose 3.
"""

from __future__ import annotations

import random
from itertools import combinations, permutations
from math import comb

from mesh.chromaticpolynomial import ChromaticPolynomial
from mesh.factories import cycle, wheel
from mesh.graph import Graph
from mesh.independencepolynomial import IndependencePolynomial
from mesh.matchingcount import PerfectMatchingCount
from mesh.probes.probe import Probe
from mesh.probes.registry import register
from mesh.triadcensus import TriadCensus
from mesh.tuttecount import TutteCount


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
def cycle_polynomial_matches_the_closed_form() -> Probe:
    # (k-1)^n + (-1)^n (k-1) at every k from zero to six
    n = 7
    cp = ChromaticPolynomial(cycle(n))
    matched = sum(1 for k in range(7) if cp.evaluate(k) == (k - 1) ** n + (-1) ** n * (k - 1))
    return Probe(
        prober="chromaticpolynomial",
        guarantee="the chromatic polynomial of a cycle follows its closed form at every k",
        holds=matched == 7,
        readings={"matched": matched, "of": 7, "colorings_with_3": cp.evaluate(3)},
    )


@register
def two_tree_counts_agree() -> Probe:
    # deletion-contraction and the matrix-tree determinant share no code
    g = wheel(6)
    tc = TutteCount(g)
    return Probe(
        prober="tuttecount",
        guarantee="deletion-contraction and Kirchhoff's determinant count the same trees",
        holds=tc.agrees_with_kirchhoff(),
        readings={"trees": tc.trees, "calls": tc.calls},
    )


@register
def matching_count_agrees_with_ryser() -> Probe:
    # the recursion on the node set and the permanent by inclusion-exclusion
    rng = random.Random(41)
    g = Graph()
    left = [f"l{i}" for i in range(6)]
    right = [f"r{i}" for i in range(6)]
    for n in left + right:
        g.add_node(n)
    for u in left:
        for v in right:
            if rng.random() < 0.5:
                g.add_edge(u, v)
    pm = PerfectMatchingCount(g)
    return Probe(
        prober="matchingcount",
        guarantee="the perfect matching recursion equals Ryser's permanent when bipartite",
        holds=pm.count == pm.permanent(),
        readings={"recursion": pm.count, "permanent": pm.permanent()},
    )


@register
def independence_polynomial_matches_enumeration() -> Probe:
    # every coefficient must equal the number of independent sets of that size
    g = _random_undirected(seed=43, n=9, p=0.35)
    ip = IndependencePolynomial(g)
    counts = [0] * (g.node_count() + 1)
    for k in range(g.node_count() + 1):
        for subset in combinations(g.nodes(), k):
            if not any(g.has_edge(a, b) for a, b in combinations(subset, 2)):
                counts[k] += 1
    while len(counts) > 1 and counts[-1] == 0:
        counts.pop()
    return Probe(
        prober="independencepolynomial",
        guarantee="each independence polynomial coefficient counts the sets of that size",
        holds=list(ip.coefficients) == counts,
        readings={"sets": ip.total_sets(), "alpha": ip.independence_number()},
    )


@register
def triads_sum_to_n_choose_three() -> Probe:
    # every triple lands in exactly one of the sixteen classes
    rng = random.Random(47)
    g = Graph(directed=True)
    names = [str(i) for i in range(9)]
    for n in names:
        g.add_node(n)
    for a, b in permutations(names, 2):
        if rng.random() < 0.3:
            g.add_edge(a, b)
    census = TriadCensus(g)
    return Probe(
        prober="triadcensus",
        guarantee="the sixteen triad counts sum to n choose 3",
        holds=census.sums_to_choose_three(),
        readings={"total": census.total(), "expected": comb(9, 3)},
    )
