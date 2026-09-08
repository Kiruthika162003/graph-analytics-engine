"""A flow day: push water through a pipe network, find the bottleneck, count the routes.

Run with: python -m examples.flowday
"""

from __future__ import annotations

from mesh.dinic import Dinic
from mesh.edmondskarp import EdmondsKarp
from mesh.graph import Graph
from mesh.menger import Menger
from mesh.mincut import MinCut
from mesh.pushrelabel import PushRelabel


def build_pipes() -> Graph:
    g = Graph(directed=True)
    for n in ["reservoir", "p1", "p2", "p3", "p4", "town"]:
        g.add_node(n)
    pipes = [
        ("reservoir", "p1", 16), ("reservoir", "p2", 13), ("p1", "p2", 10),
        ("p2", "p1", 4), ("p1", "p3", 12), ("p3", "p2", 9), ("p2", "p4", 14),
        ("p4", "p3", 7), ("p3", "town", 20), ("p4", "town", 4),
    ]
    for a, b, litres in pipes:
        g.add_edge(a, b, litres)
    return g


def main() -> int:
    g = build_pipes()
    ek = EdmondsKarp(g, "reservoir", "town")
    dn = Dinic(g, "reservoir", "town")
    pr = PushRelabel(g, "reservoir", "town")
    print(f"max flow by three algorithms: {ek.value}, {dn.value}, {pr.value}")
    print(f"  edmonds-karp {ek.augmentations} augmentations, dinic {dn.phases} phases")
    print(f"  push-relabel {pr.pushes} pushes and {pr.relabels} relabels")

    cut = MinCut(g, "reservoir", "town")
    print(f"bottleneck: {len(cut.edges)} pipe(s) with capacity {cut.capacity()}")
    for u, v, cap in cut.edges:
        print(f"  {u} -> {v} at {cap}, saturated: {cut.every_cut_edge_is_saturated()}")

    routes = Menger(g, "reservoir", "town")
    print(f"independent routes: {routes.edge_disjoint} by pipe")
    print(f"  and {routes.node_disjoint} by junction")
    print(routes.note())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
