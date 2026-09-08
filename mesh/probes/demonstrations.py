"""The probe demonstrations: each function builds a graph and takes readings.

Every probe here is registered into the registry on import. A probe returns
a Probe with a boolean that must stay true and the numbers behind it.
"""

from __future__ import annotations

from mesh.graph import Graph
from mesh.probes.probe import Probe
from mesh.probes.registry import register


@register
def handshake_lemma() -> Probe:
    # the sum of degrees over all nodes equals twice the edge count, always,
    # because each undirected edge contributes one to each of its endpoints
    g = Graph(directed=False)
    for node in "abcdef":
        g.add_node(node)
    for u, v in [("a", "b"), ("a", "c"), ("b", "c"), ("c", "d"), ("d", "e")]:
        g.add_edge(u, v)
    degree_sum = sum(g.degree(n) for n in g.nodes())
    twice_edges = 2 * g.edge_count()
    return Probe(
        prober="handshake",
        guarantee=(
            "the degree sum equals twice the edge count, so no edge is counted "
            "at only one end"
        ),
        holds=degree_sum == twice_edges,
        readings={"degree_sum": degree_sum, "twice_edges": twice_edges},
    )
