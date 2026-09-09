"""Spectral and logic probes: eigenvalues held to identities, two decisions to brute force.

A cospectral pair that is not isomorphic, the Laplacian tree count
against Kirchhoff, the Cheeger bracket, 2-SAT against every
assignment, and edit distance zero on a relabeled graph.
"""

from __future__ import annotations

import random
from itertools import combinations, product

from mesh.cheeger import Cheeger
from mesh.editdistance import EditDistance
from mesh.factories import cycle, star
from mesh.graph import Graph
from mesh.graphenergy import GraphEnergy
from mesh.isomorphism import Isomorphism
from mesh.laplacianspectrum import LaplacianSpectrum
from mesh.probes.probe import Probe
from mesh.probes.registry import register
from mesh.twosat import Clause, TwoSat


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
def cospectral_mates_are_not_isomorphic() -> Probe:
    # the square plus a lone node and the four-leaf star share an adjacency spectrum
    square = cycle(4, prefix="c")
    square.add_node("lone")
    claw = star(4, prefix="s")
    a = [round(x, 6) for x in GraphEnergy(square).spectrum]
    b = [round(x, 6) for x in GraphEnergy(claw).spectrum]
    same_spectrum = a == b
    same_shape = Isomorphism(square, claw).isomorphic
    return Probe(
        prober="graphenergy",
        guarantee="a square plus a lone node and the claw are cospectral but not isomorphic",
        holds=same_spectrum and not same_shape,
        readings={"spectrum": a, "isomorphic": same_shape},
    )


@register
def laplacian_trees_match_kirchhoff() -> Probe:
    # the product of non-zero Laplacian eigenvalues over n is the spanning tree count
    g = _random_undirected(seed=53, n=8, p=0.5)
    ls = LaplacianSpectrum(g)
    return Probe(
        prober="laplacianspectrum",
        guarantee="the non-zero Laplacian eigenvalues multiply to n times the tree count",
        holds=ls.agrees_with_kirchhoff() and ls.trace_identity_holds(),
        readings={"trees": round(ls.spanning_trees()), "lambda2": round(ls.values[1], 6)},
    )


@register
def cheeger_bracket_holds() -> Probe:
    # lambda2 / 2 <= h <= sqrt(2 lambda2) on a connected random graph
    g = _random_undirected(seed=59, n=9, p=0.4)
    for a, b in zip(g.nodes(), g.nodes()[1:], strict=False):
        if not g.has_edge(a, b):
            g.add_edge(a, b)
    ch = Cheeger(g)
    _side, h = ch.exact()
    return Probe(
        prober="cheeger",
        guarantee="the Cheeger constant sits between lambda2 over 2 and root 2 lambda2",
        holds=ch.bounds_hold(),
        readings={"h": round(h, 6), "lambda2": round(ch.lambda2, 6)},
    )


@register
def two_sat_matches_every_assignment() -> Probe:
    # the component verdict must equal a search over all 2^n assignments
    rng = random.Random(61)
    names = [f"v{i}" for i in range(6)]
    agreed = 0
    total = 20
    for _ in range(total):
        clauses: list[Clause] = [
            ((rng.choice(names), rng.random() < 0.5), (rng.choice(names), rng.random() < 0.5))
            for _ in range(10)
        ]
        ts = TwoSat(clauses)
        brute = any(
            all(vals[a[0]] == a[1] or vals[b[0]] == b[1] for a, b in clauses)
            for vals in (
                dict(zip(names, bits, strict=True))
                for bits in product((False, True), repeat=len(names))
            )
        )
        if ts.satisfiable == brute and (not ts.satisfiable or ts.satisfies(ts.assignment())):
            agreed += 1
    return Probe(
        prober="twosat",
        guarantee="the component verdict of 2-SAT equals trying every assignment",
        holds=agreed == total,
        readings={"agreed": agreed, "instances": total},
    )


@register
def relabeled_graph_has_zero_edit_distance() -> Probe:
    # edit distance is zero exactly when the graphs are the same shape
    g = _random_undirected(seed=67, n=6, p=0.5)
    rng = random.Random(68)
    names = g.nodes()
    shuffled = list(names)
    rng.shuffle(shuffled)
    mapping = dict(zip(names, shuffled, strict=True))
    h = Graph()
    for n in shuffled:
        h.add_node(n)
    for u, v, _w in g.edges():
        h.add_edge(mapping[u], mapping[v])
    ed = EditDistance(g, h)
    return Probe(
        prober="editdistance",
        guarantee="a relabeled copy is at edit distance zero with every node matched",
        holds=ed.distance == 0 and len(ed.matched_pairs()) == g.node_count(),
        readings={"distance": ed.distance, "matched": len(ed.matched_pairs())},
    )
