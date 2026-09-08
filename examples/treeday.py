"""A tree day: one org chart, four ways to flatten it into fast queries.

Run with: python -m examples.treeday
"""

from __future__ import annotations

from mesh.centroid import CentroidDecomposition
from mesh.eulertour import EulerTour
from mesh.graph import Graph
from mesh.hld import HeavyLight
from mesh.lca import LowestCommonAncestor


def build_org() -> tuple[Graph, dict[str, float]]:
    g = Graph()
    reports = [
        ("ceo", "cto"), ("ceo", "cfo"), ("cto", "platform"), ("cto", "mobile"),
        ("platform", "infra"), ("platform", "data"), ("infra", "oncall"),
        ("mobile", "ios"), ("mobile", "android"), ("cfo", "payroll"),
    ]
    for a, b in reports:
        g.add_node(a)
        g.add_node(b)
        g.add_edge(a, b)
    headcount = {
        "ceo": 1, "cto": 2, "cfo": 2, "platform": 5, "mobile": 4, "infra": 6,
        "data": 3, "oncall": 8, "ios": 7, "android": 7, "payroll": 3,
    }
    return g, {k: float(v) for k, v in headcount.items()}


def main() -> int:
    g, heads = build_org()
    print(f"people: {g.node_count()}, reporting lines: {g.edge_count()}")

    lca = LowestCommonAncestor(g, "ceo")
    boss = lca.lca("oncall", "data")
    apart = lca.distance("oncall", "data")
    print(f"first shared manager of oncall and data: {boss}, {apart} hops apart")

    hld = HeavyLight(g, "ceo", heads)
    biggest = hld.path_max("oncall", "ios")
    print(f"largest team on the chain from oncall to ios: {biggest:.0f}")
    print(hld.note())

    tour = EulerTour(g, "ceo", heads)
    print(f"cto's org spans {tour.subtree_size('cto')} nodes")
    print(f"  largest team under cto: {tour.subtree_max('cto'):.0f}")
    print(f"is platform under cto: {tour.is_ancestor('cto', 'platform')}")

    cd = CentroidDecomposition(g)
    print(cd.note())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
