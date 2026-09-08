"""A centrality day: seven notions of importance, ranked side by side on one network.

Run with: python -m examples.centralityday
"""

from __future__ import annotations

from mesh.betweenness import Betweenness
from mesh.closeness import Closeness
from mesh.eigenvector import EigenvectorCentrality
from mesh.graph import Graph
from mesh.hits import HITS
from mesh.katz import Katz
from mesh.pagerank import PageRank


def build_office() -> Graph:
    # two teams, a liaison between them, and a well-liked newcomer
    g = Graph()
    edges = [
        ("mira", "otto"), ("mira", "pam"), ("otto", "pam"), ("pam", "quinn"),
        ("quinn", "rosa"), ("quinn", "sol"), ("rosa", "sol"), ("rosa", "tess"),
        ("sol", "tess"), ("tess", "uma"), ("mira", "uma"),
    ]
    for a, b in edges:
        g.add_node(a)
        g.add_node(b)
        g.add_edge(a, b)
    return g


def to_directed(g: Graph) -> Graph:
    d = Graph(directed=True)
    for n in g.nodes():
        d.add_node(n)
    for u, v, w in g.edges():
        d.add_edge(u, v, w)
        d.add_edge(v, u, w)
    return d


def main() -> int:
    g = build_office()
    print(f"people: {g.node_count()}, ties: {g.edge_count()}")
    degree = sorted(g.nodes(), key=lambda n: (-g.degree(n), n))[0]
    print(f"degree:      {degree}")
    print(f"closeness:   {Closeness(g).top(1)[0][0]}")
    print(f"betweenness: {Betweenness(g).top(1)[0][0]}")
    print(f"eigenvector: {EigenvectorCentrality(g).top(1)[0][0]}")
    print(f"katz:        {Katz(g, alpha=0.15).top(1)[0][0]}")
    d = to_directed(g)
    print(f"pagerank:    {PageRank(d).top(1)[0][0]}")
    h = HITS(d)
    print(f"hits hub:    {h.top_hubs(1)[0][0]}, authority: {h.top_authorities(1)[0][0]}")
    print(Betweenness(g).note())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
