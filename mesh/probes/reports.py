"""Report probes: the self-check, the full report, and partition agreement held to their frames.

The self-check fails nothing on a random graph, the full report writes
or skips every section and counts them, and identical partitions agree
completely while singletons against one group share no information.
"""

from __future__ import annotations

import random
from itertools import combinations

from mesh.graph import Graph
from mesh.graphcheck import SelfCheck
from mesh.graphreport import SECTIONS, FullReport
from mesh.partitioncompare import PartitionCompare
from mesh.probes.probe import Probe
from mesh.probes.registry import register


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
def self_check_fails_nothing_on_a_random_graph() -> Probe:
    # every universal identity must hold on a graph nobody built to satisfy it
    sc = SelfCheck(_random_undirected(seed=157, n=10, p=0.4))
    return Probe(
        prober="graphcheck",
        guarantee="the self-check's universal identities all hold on a random graph",
        holds=sc.all_held(),
        readings=sc.counts(),
    )


@register
def full_report_accounts_for_every_section() -> Probe:
    # written plus skipped must equal the number of sections, on two very different inputs
    plain = FullReport(_random_undirected(seed=163, n=8, p=0.4))
    empty = FullReport(Graph())
    return Probe(
        prober="graphreport",
        guarantee="the full report writes or skips every section and its tally adds up",
        holds=plain.written + plain.skipped == len(SECTIONS)
        and empty.written + empty.skipped == len(SECTIONS),
        readings={"written": plain.written, "skipped": plain.skipped, "of": len(SECTIONS)},
    )


@register
def partition_agreement_reads_its_ends() -> Probe:
    # identical partitions score one; singletons against one group score zero information
    g = _random_undirected(seed=167, n=9, p=0.4)
    groups = [g.nodes()[:4], g.nodes()[4:]]
    same = PartitionCompare(g, groups, list(reversed(groups)))
    apart = PartitionCompare(g, [[n] for n in g.nodes()], [g.nodes()])
    return Probe(
        prober="partitioncompare",
        guarantee="identical partitions agree fully; singletons against one group share none",
        holds=abs(same.nmi() - 1.0) < 1e-9 and same.rand() == 1.0 and abs(apart.nmi()) < 1e-9,
        readings={"same_nmi": round(same.nmi(), 6), "apart_nmi": round(apart.nmi(), 6)},
    )
