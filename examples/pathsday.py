"""A paths day: six ways to ask for a route on one map, and when they disagree.

Run with: python -m examples.pathsday
"""

from __future__ import annotations

from mesh.astar import AStar
from mesh.bidijkstra import BidirectionalDijkstra
from mesh.dijkstra import Dijkstra
from mesh.graph import Graph
from mesh.widestpath import WidestPath
from mesh.yen import YenKShortest


def build_map() -> tuple[Graph, dict[str, tuple[int, int]]]:
    # a small town grid with a river crossing that is short but narrow
    g = Graph()
    coords = {
        "depot": (0, 0), "mill": (2, 0), "bridge": (4, 0), "market": (6, 0),
        "farm": (0, 3), "chapel": (3, 3), "ford": (6, 3),
    }
    for name in coords:
        g.add_node(name)
    roads = [
        ("depot", "mill", 2), ("mill", "bridge", 2), ("bridge", "market", 2),
        ("depot", "farm", 3), ("farm", "chapel", 3), ("chapel", "ford", 3),
        ("ford", "market", 3), ("mill", "chapel", 4), ("chapel", "bridge", 2),
    ]
    for a, b, km in roads:
        g.add_edge(a, b, km)
    return g, coords


def main() -> int:
    g, coords = build_map()
    print(f"places: {g.node_count()}, roads: {g.edge_count()}")

    dij = Dijkstra(g, "depot")
    print(f"dijkstra depot to market: {dij.distance_to('market')} km")
    print(f"  via {dij.path_to('market')}")

    mx, my = coords["market"]

    def manhattan(node: str) -> float:
        x, y = coords[node]
        return abs(x - mx) + abs(y - my)

    star = AStar(g, "depot", "market", heuristic=manhattan)
    print(f"a-star agrees at {star.cost()} km, expanding {star.expanded} of {g.node_count()}")

    both = BidirectionalDijkstra(g, "depot", "market")
    print(f"bidirectional agrees at {both.distance()} km, settling {both.settled}")

    wide = WidestPath(g, "depot")
    print(f"widest road to market carries {wide.width_to('market')}")
    print(f"  via {wide.path_to('market')}")

    yen = YenKShortest(g, "depot", "market", 3)
    print(f"three shortest routes cost {yen.costs()}")
    print(yen.note())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
