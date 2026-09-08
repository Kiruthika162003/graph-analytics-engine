"""A cut day: every pair's bottleneck from one tree, and a flow taken apart into routes.

Run with: python -m examples.cutday
"""

from __future__ import annotations

from itertools import combinations

from mesh.edgecover import EdgeCover
from mesh.flowdecomposition import FlowDecomposition
from mesh.gomoryhu import GomoryHu
from mesh.graph import Graph


def build_pipes() -> Graph:
    g = Graph()
    pipes = [
        ("well", "tank", 8.0), ("well", "pump", 5.0), ("tank", "pump", 3.0),
        ("tank", "north", 6.0), ("pump", "south", 7.0), ("north", "south", 2.0),
        ("north", "town", 5.0), ("south", "town", 6.0),
    ]
    for a, b, w in pipes:
        g.add_node(a)
        g.add_node(b)
        g.add_edge(a, b, w)
    return g


def main() -> int:
    g = build_pipes()
    gh = GomoryHu(g)
    print(gh.note())
    for a, b in combinations(["well", "tank", "north", "town"], 2):
        print(f"  bottleneck {a}-{b}: {gh.min_cut(a, b):.0f}")

    fd = FlowDecomposition(gh.flow_graph, "well", "town")
    print(fd.note())
    for trail, amount in fd.paths:
        print(f"  {amount:.0f} along {' > '.join(trail)}")

    ec = EdgeCover(g)
    print(ec.note())
    print(f"  cover: {', '.join(f'{u}-{v}' for u, v in ec.cover)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
