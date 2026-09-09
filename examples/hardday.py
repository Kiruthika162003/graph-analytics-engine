"""A hard day: cycles to break, a line to lay out, a fire to spread, and a tour to drive.

Run with: python -m examples.hardday
"""

from __future__ import annotations

from math import hypot

from mesh.burning import Burning
from mesh.centralization import Centralization
from mesh.factories import cycle, grid, path, wheel
from mesh.feedbackvertexset import FeedbackVertexSet
from mesh.graph import Graph
from mesh.graphgrammar import Rewriter, Rule
from mesh.pathwidth import Pathwidth
from mesh.tsp import TravellingSalesman


def build_towns() -> Graph:
    spots = {
        "mill": (0.0, 0.0), "forge": (4.0, 1.0), "dock": (6.0, 5.0),
        "farm": (2.0, 6.0), "inn": (1.0, 3.0), "kiln": (5.0, 3.0),
    }
    g = Graph()
    for name in spots:
        g.add_node(name)
    names = list(spots)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            (xa, ya), (xb, yb) = spots[a], spots[b]
            g.add_edge(a, b, round(hypot(xa - xb, ya - yb), 2))
    return g


def main() -> int:
    for name, g in (("wheel", wheel(6)), ("grid", grid(3)), ("cycle", cycle(8))):
        print(f"{name:<6} {FeedbackVertexSet(g).note()}")
        print(f"{'':<6} {Pathwidth(g).note()}")
        print(f"{'':<6} {Burning(g).note()}")

    contract = Rule(_pattern("a b c", [("a", "b"), ("b", "c")]), _pattern("a c", [("a", "c")]))
    rw = Rewriter(path(7), contract)
    rw.apply_until_fixed()
    print(f"contracting a path of seven: {rw.note()}")

    print(f"towns: {TravellingSalesman(build_towns()).note()}")
    print(f"towns: {Centralization(build_towns()).note()}")
    return 0


def _pattern(nodes: str, edges: list[tuple[str, str]]) -> Graph:
    g = Graph()
    for n in nodes.split():
        g.add_node(n)
    for u, v in edges:
        g.add_edge(u, v)
    return g


if __name__ == "__main__":
    raise SystemExit(main())
