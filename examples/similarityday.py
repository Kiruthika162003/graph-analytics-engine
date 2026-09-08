"""A similarity day: who is like whom, which link comes next, and is this the same graph.

Run with: python -m examples.similarityday
"""

from __future__ import annotations

from mesh.factories import cycle, petersen
from mesh.graph import Graph
from mesh.isomorphism import Isomorphism
from mesh.linkprediction import LinkPrediction
from mesh.simrank import SimRank
from mesh.wlhash import WLHash


def build_readers() -> Graph:
    g = Graph()
    ties = [
        ("ada", "ben"), ("ada", "cal"), ("ben", "cal"), ("cal", "dee"),
        ("dee", "eve"), ("dee", "fox"), ("eve", "fox"), ("fox", "ada"),
    ]
    for a, b in ties:
        g.add_node(a)
        g.add_node(b)
        g.add_edge(a, b)
    return g


def main() -> int:
    g = build_readers()
    print(f"readers: {g.node_count()}, ties: {g.edge_count()}")

    lp = LinkPrediction(g)
    for measure in ("common", "jaccard", "adamic_adar", "preferential"):
        u, v, score = lp.ranked(measure)[0]
        print(f"next link by {measure:<12} {u}-{v} at {score:.2f}")

    directed = Graph(directed=True)
    for n in g.nodes():
        directed.add_node(n)
    for u, v, _w in g.edges():
        directed.add_edge(u, v)
        directed.add_edge(v, u)
    sr = SimRank(directed)
    print(f"simrank: ada's closest match is {sr.most_similar('ada', 1)[0][0]}")

    a, b = cycle(10, prefix="c"), petersen()
    same_hash = WLHash(a).digest == WLHash(b).digest
    same_shape = Isomorphism(a, b).isomorphic
    print(f"10-cycle vs petersen: hashes equal {same_hash}, isomorphic {same_shape}")
    print(WLHash(b).note())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
