"""A first walk: build a small road network and ask it the basic questions.

Run with: python -m examples.firstwalk
"""

from __future__ import annotations

from mesh.bfs import BFS
from mesh.connectedcomponents import ConnectedComponents
from mesh.diameter import Diameter
from mesh.dijkstra import Dijkstra
from mesh.graph import Graph


def build_towns() -> Graph:
    g = Graph()
    for town in ["ash", "birch", "cedar", "dune", "elm", "fern", "grove"]:
        g.add_node(town)
    roads = [
        ("ash", "birch", 4), ("ash", "cedar", 2), ("birch", "cedar", 5),
        ("birch", "dune", 10), ("cedar", "dune", 3), ("dune", "elm", 4),
        ("elm", "fern", 11), ("fern", "grove", 2),
    ]
    for a, b, km in roads:
        g.add_edge(a, b, km)
    return g


def main() -> int:
    g = build_towns()
    print(f"towns: {g.node_count()}, roads: {g.edge_count()}")

    hops = BFS(g, "ash")
    print(f"hops from ash to grove: {hops.distance_to('grove')} via {hops.path_to('grove')}")

    km = Dijkstra(g, "ash")
    print(f"shortest drive ash to grove: {km.distance_to('grove')} km")
    print(f"  via {km.path_to('grove')}")
    # the fewest hops and the shortest drive disagree when a long road skips towns
    differ = hops.path_to("dune") != km.path_to("dune")
    print(f"hops and kilometres pick different routes to dune: {differ}")

    parts = ConnectedComponents(g)
    print(f"components: {parts.count()}, giant share {parts.giant_share():.0%}")

    span = Diameter(g)
    print(f"diameter {span.diameter()} hops, center {sorted(span.center())}")
    print(span.note())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
