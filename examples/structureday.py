"""A structure day: where one failure splits a network, and what it would take to fix.

Run with: python -m examples.structureday
"""

from __future__ import annotations

from itertools import combinations

from mesh.articulation import Articulation
from mesh.biconnected import Biconnected
from mesh.connectivitynumbers import ConnectivityNumbers
from mesh.graph import Graph
from mesh.kcore import KCore
from mesh.twoedgeconnected import TwoEdgeConnected


def build_campus() -> Graph:
    # two dense buildings joined by one link through a gateway switch
    g = Graph()
    north = ["n1", "n2", "n3", "n4"]
    south = ["s1", "s2", "s3", "s4"]
    for n in north + south + ["gateway", "annex"]:
        g.add_node(n)
    for a, b in combinations(north, 2):
        g.add_edge(a, b)
    for a, b in combinations(south, 2):
        g.add_edge(a, b)
    g.add_edge("n4", "gateway")
    g.add_edge("gateway", "s1")
    g.add_edge("s4", "annex")
    return g


def main() -> int:
    g = build_campus()
    print(f"switches: {g.node_count()}, links: {g.edge_count()}")

    weak = Articulation(g)
    print(f"single points of failure: {sorted(weak.points)}")
    print(f"single links of failure: {sorted(sorted(b) for b in weak.bridges)}")

    tec = TwoEdgeConnected(g)
    print(f"pieces that survive any one cut link: {len(tec.components)}")
    print(f"links to add for full redundancy: {tec.edges_to_add()}")

    blocks = Biconnected(g)
    print(f"blocks: {len(blocks.blocks)} glued at {sorted(blocks.cut_nodes)}")

    core = KCore(g)
    print(f"innermost core: {sorted(core.k_core(core.degeneracy()))} at k={core.degeneracy()}")

    numbers = ConnectivityNumbers(g)
    print(numbers.note())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
