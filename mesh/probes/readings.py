"""Reading probes: centralities, walks, and shapes held to identities that need no trust.

Stress against a direct pair count, commute time against resistance,
graphlet orbit totals, the common subgraph of a relabeled copy, and
the Wiener index of a tree against its edge cuts.
"""

from __future__ import annotations

import random
from itertools import combinations

from mesh.commonsubgraph import CommonSubgraph
from mesh.commutetime import CommuteTime
from mesh.factories import path
from mesh.graph import Graph
from mesh.graphlets import Graphlets
from mesh.probes.probe import Probe
from mesh.probes.registry import register
from mesh.prufer import Prufer
from mesh.stresscentrality import StressCentrality
from mesh.wienerindex import WienerIndex


def _random_connected(seed: int, n: int, p: float) -> Graph:
    rng = random.Random(seed)
    g = path(n)
    for a, b in combinations(g.nodes(), 2):
        if not g.has_edge(a, b) and rng.random() < p:
            g.add_edge(a, b)
    return g


@register
def stress_of_a_path_is_the_side_product() -> Probe:
    # inner node i of a path on n has i nodes on one side and n-1-i on the other
    n = 9
    sc = StressCentrality(path(n))
    matched = sum(1 for i in range(n) if sc.stress[str(i)] == i * (n - 1 - i))
    return Probe(
        prober="stresscentrality",
        guarantee="a path node's stress is the product of the node counts on its two sides",
        holds=matched == n,
        readings={"matched": matched, "nodes": n},
    )


@register
def commute_time_is_twice_edges_times_resistance() -> Probe:
    # the walk and the circuit must agree on three pairs of a random connected graph
    g = _random_connected(seed=71, n=8, p=0.3)
    ct = CommuteTime(g)
    pairs = [("0", "7"), ("2", "5"), ("1", "6")]
    matched = sum(1 for a, b in pairs if ct.commute_matches_resistance(a, b))
    return Probe(
        prober="commutetime",
        guarantee="commute time equals twice the edge count times the effective resistance",
        holds=matched == len(pairs),
        readings={"matched": matched, "pairs": len(pairs), "edges": g.edge_count()},
    )


@register
def graphlet_orbits_obey_their_totals() -> Probe:
    # ends twice the paths, middles once, corners three times the triangles
    g = _random_connected(seed=73, n=10, p=0.35)
    gl = Graphlets(g)
    return Probe(
        prober="graphlets",
        guarantee="orbit totals count each induced path and triangle the right number of times",
        holds=gl.identities_hold(),
        readings={"paths": gl.paths, "triangles": gl.triangles},
    )


@register
def relabeled_copy_is_fully_common() -> Probe:
    # a graph and a relabeling of it share every node and edge
    g = _random_connected(seed=79, n=6, p=0.4)
    h = Graph()
    for n in g.nodes():
        h.add_node("r" + n)
    for u, v, _w in g.edges():
        h.add_edge("r" + u, "r" + v)
    cs = CommonSubgraph(g, h)
    return Probe(
        prober="commonsubgraph",
        guarantee="a graph and its relabeling have all nodes and edges in common",
        holds=cs.size() == g.node_count() and cs.shared_edges() == g.edge_count(),
        readings={"common": cs.size(), "edges": cs.shared_edges()},
    )


@register
def tree_wiener_index_is_the_edge_cut_sum() -> Probe:
    # each tree edge is crossed by exactly the pairs it separates
    rng = random.Random(83)
    tree = Prufer.decode([rng.randrange(10) for _ in range(8)])
    wi = WienerIndex(tree)
    return Probe(
        prober="wienerindex",
        guarantee="a tree's Wiener index is the sum over edges of the product of its two sides",
        holds=wi.tree_edge_identity_holds(),
        readings={"wiener": wi.wiener(), "nodes": tree.node_count()},
    )
