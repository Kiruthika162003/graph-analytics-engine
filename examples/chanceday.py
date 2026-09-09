"""A chance day: shuffles against the real graph, edges that fail, and a rumor that spreads.

Run with: python -m examples.chanceday
"""

from __future__ import annotations

from itertools import combinations

from mesh.epidemic import Epidemic
from mesh.factories import cycle
from mesh.graph import Graph
from mesh.nullmodel import NullModel
from mesh.percolation import Percolation


def build_clubs() -> Graph:
    g = Graph()
    clubs = [["ada", "ben", "cal", "dee"], ["eve", "fox", "gus", "hal"], ["ivy", "jon", "kim"]]
    for club in clubs:
        for n in club:
            g.add_node(n)
        for a, b in combinations(club, 2):
            g.add_edge(a, b)
    for a, b in [("dee", "eve"), ("hal", "ivy"), ("kim", "ada")]:
        g.add_edge(a, b)
    return g


def main() -> int:
    clubs = build_clubs()
    print(f"clubs: {NullModel(clubs, samples=20, seed=1).note()}")
    print(f"ring : {NullModel(cycle(12), samples=20, seed=1).note()}")

    print(f"clubs: {Percolation(clubs, seed=2).note(steps=5, trials=20)}")

    for beta in (0.1, 0.3, 0.6):
        print(f"clubs: {Epidemic(clubs, beta=beta, gamma=0.5, seed=3).note(trials=20)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
