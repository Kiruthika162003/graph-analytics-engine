"""A recipe day: four questions asked of one network, each answered with its steps shown.

Run with: python -m examples.recipeday
"""

from __future__ import annotations

from itertools import combinations

from mesh.graph import Graph
from mesh.graphdiffreport import ChangeReport
from mesh.graphrecipes import Recipes


def build_departments() -> Graph:
    g = Graph()
    teams = {
        "design": ["ada", "ben", "cal", "dee"],
        "build": ["eve", "fox", "gus", "hal", "ivy"],
        "sales": ["jon", "kim", "lee"],
    }
    for members in teams.values():
        for n in members:
            g.add_node(n)
        for a, b in combinations(members, 2):
            g.add_edge(a, b)
    for a, b in [("dee", "eve"), ("ivy", "jon"), ("cal", "kim")]:
        g.add_edge(a, b)
    return g


def main() -> int:
    g = build_departments()
    recipes = Recipes(g, seed=1)
    for name in ("communities", "bottleneck", "robustness", "influence"):
        print(f"== {name}")
        for line in recipes.run(name):
            print(f"  {line}")

    later = build_departments()
    later.add_node("max")
    later.add_edge("max", "ada")
    later.add_edge("max", "eve")
    later.add_edge("max", "jon")
    print("== a quarter later")
    for line in ChangeReport(g, later).lines():
        print(f"  {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
