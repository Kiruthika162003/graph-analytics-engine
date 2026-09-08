"""Class probes: graph classes, cut trees, products, and entropy held to their identities.

Chordal clique numbers against Bron-Kerbosch, threshold graphs against
split-and-cograph, the Gomory-Hu tree against direct flows, Cartesian
product distances against coordinate sums, and structural entropy
under the two partitions that must agree.
"""

from __future__ import annotations

import random
from collections import deque
from itertools import combinations

from mesh.bronkerbosch import BronKerbosch
from mesh.chordal import Chordal
from mesh.entropy import GraphEntropy
from mesh.factories import cycle, path
from mesh.gomoryhu import GomoryHu
from mesh.graph import Graph
from mesh.graphproduct import GraphProduct
from mesh.mincut import MinCut
from mesh.probes.probe import Probe
from mesh.probes.registry import register
from mesh.thresholdgraph import ThresholdGraph


def _random_undirected(seed: int, n: int, p: float, weighted: bool = False) -> Graph:
    rng = random.Random(seed)
    g = Graph()
    nodes = [str(i) for i in range(n)]
    for node in nodes:
        g.add_node(node)
    for a, b in combinations(nodes, 2):
        if rng.random() < p:
            g.add_edge(a, b, float(rng.randint(1, 9)) if weighted else 1.0)
    return g


def _hops(g: Graph, start: str) -> dict[str, int]:
    dist = {start: 0}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for other in g.neighbors(node):
            if other not in dist:
                dist[other] = dist[node] + 1
                queue.append(other)
    return dist


@register
def chordal_clique_matches_bron_kerbosch() -> Probe:
    # a fan is chordal, and its elimination ordering must read the exact clique number
    g = path(7)
    g.add_node("hub")
    for i in range(7):
        g.add_edge("hub", str(i))
    c = Chordal(g)
    exact = BronKerbosch(g).clique_number()
    return Probe(
        prober="chordal",
        guarantee="on a chordal graph the elimination ordering reads the exact clique number",
        holds=c.is_chordal and c.clique_number() == exact,
        readings={"ordering": c.clique_number(), "bronkerbosch": exact},
    )


@register
def threshold_is_split_and_cograph() -> Probe:
    # the three recognizers must agree on every random graph in the sample
    agreed = 0
    total = 24
    for seed in range(total):
        g = _random_undirected(seed=600 + seed, n=7, p=0.5)
        if ThresholdGraph(g).matches_split_and_cograph():
            agreed += 1
    return Probe(
        prober="thresholdgraph",
        guarantee="a graph is threshold exactly when it is both split and a cograph",
        holds=agreed == total,
        readings={"agreed": agreed, "graphs": total},
    )


@register
def cut_tree_agrees_with_every_flow() -> Probe:
    # every pair's tree answer must equal a direct flow computation
    g = _random_undirected(seed=31, n=8, p=0.5, weighted=True)
    gh = GomoryHu(g)
    pairs = list(combinations(g.nodes(), 2))
    flows = gh.flow_graph
    matched = sum(1 for a, b in pairs if gh.min_cut(a, b) == MinCut(flows, a, b).value)
    return Probe(
        prober="gomoryhu",
        guarantee="the Gomory-Hu tree answers every pairwise minimum cut like a direct flow",
        holds=matched == len(pairs),
        readings={"matched": matched, "pairs": len(pairs), "flows": gh.flows},
    )


@register
def cartesian_distances_add() -> Probe:
    # a hop count in the product must be the sum of the coordinate hop counts
    a, b = path(4), cycle(5)
    prod = GraphProduct(a, b).cartesian()
    da, db, dp = _hops(a, "0"), _hops(b, "0"), _hops(prod, "0,0")
    checked = sum(1 for x in a.nodes() for y in b.nodes() if dp[f"{x},{y}"] == da[x] + db[y])
    return Probe(
        prober="graphproduct",
        guarantee="distances in a Cartesian product are the sum of the coordinate distances",
        holds=checked == prod.node_count(),
        readings={"checked": checked, "nodes": prod.node_count()},
    )


@register
def structural_entropy_partitions_agree() -> Probe:
    # the trivial partition and the singleton partition score the same bits
    g = _random_undirected(seed=37, n=9, p=0.4)
    e = GraphEntropy(g)
    whole = e.partition_entropy([g.nodes()])
    singles = e.partition_entropy([[n] for n in g.nodes()])
    return Probe(
        prober="entropy",
        guarantee="the trivial and singleton partitions both score the one-dimensional entropy",
        holds=abs(whole - singles) < 1e-9 and abs(whole - e.structural_entropy()) < 1e-9,
        readings={"whole": round(whole, 6), "singles": round(singles, 6)},
    )
