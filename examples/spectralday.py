"""A spectral day: eigenvalues by rotation, energy, the Fiedler cut, and the Cheeger bracket.

Run with: python -m examples.spectralday
"""

from __future__ import annotations

from itertools import combinations

from mesh.cheeger import Cheeger
from mesh.factories import complete, cycle, path, star
from mesh.graph import Graph
from mesh.graphenergy import GraphEnergy
from mesh.laplacianspectrum import LaplacianSpectrum
from mesh.symmetriceigen import SymmetricEigen


def build_two_teams() -> Graph:
    g = Graph()
    for team in (["ada", "ben", "cal", "dee"], ["eve", "fox", "gus", "hal"]):
        for n in team:
            g.add_node(n)
        for a, b in combinations(team, 2):
            g.add_edge(a, b)
    g.add_edge("dee", "eve")
    g.add_edge("cal", "fox")
    return g


def main() -> int:
    se = SymmetricEigen([[2.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 2.0]])
    print(f"tridiagonal: {se.note()}")

    shapes = (("path", path(6)), ("cycle", cycle(6)), ("star", star(5)), ("K5", complete(5)))
    for name, g in shapes:
        print(f"{name:<6} {GraphEnergy(g).note()}")

    teams = build_two_teams()
    ls = LaplacianSpectrum(teams)
    print(f"two teams: {ls.note()}")
    left, right = ls.fiedler_partition()
    print(f"  fiedler puts {left} against {right}")

    ch = Cheeger(teams)
    print(f"two teams: {ch.note()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
