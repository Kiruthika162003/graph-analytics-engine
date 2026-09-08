"""A community day: three detectors on one club network, and what modularity says.

Run with: python -m examples.communityday
"""

from __future__ import annotations

from itertools import combinations

from mesh.girvannewman import GirvanNewman
from mesh.graph import Graph
from mesh.labelpropagation import LabelPropagation
from mesh.louvain import Louvain
from mesh.triangles import Triangles


def build_clubs() -> Graph:
    g = Graph()
    clubs = {
        "chess": ["ana", "bo", "cy", "dev"],
        "rowing": ["eli", "fay", "gus", "hal"],
        "choir": ["ivy", "jo", "kim", "lee"],
    }
    for members in clubs.values():
        for m in members:
            g.add_node(m)
        for a, b in combinations(members, 2):
            g.add_edge(a, b)
    # a few people belong to two worlds
    g.add_edge("dev", "eli")
    g.add_edge("hal", "ivy")
    g.add_edge("ana", "lee")
    return g


def main() -> int:
    g = build_clubs()
    print(f"people: {g.node_count()}, friendships: {g.edge_count()}")
    print(Triangles(g).note())

    lp = LabelPropagation(g, seed=3)
    print(f"label propagation: {len(lp.communities())} groups")
    print(f"  modularity {lp.modularity():.3f}")

    lv = Louvain(g, seed=3)
    print(f"louvain: {len(lv.communities())} groups, modularity {lv.modularity():.3f}")
    for group in lv.communities():
        print(f"  {sorted(group)}")

    gn = GirvanNewman(g)
    print(f"girvan-newman: peak modularity {gn.best_modularity:.3f} after {gn.best_at} cuts")
    print(f"  first edges cut: {gn.removed[: gn.best_at]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
