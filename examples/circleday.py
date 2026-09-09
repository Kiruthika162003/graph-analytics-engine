"""A circle day: who brokers, how walks pile up, and whether two groupings agree.

Run with: python -m examples.circleday
"""

from __future__ import annotations

from itertools import combinations

from mesh.graph import Graph
from mesh.neighborhood import Neighborhood
from mesh.partitioncompare import PartitionCompare
from mesh.spectralclustering import SpectralClustering
from mesh.walkcount import WalkCount


def build_circles() -> Graph:
    g = Graph()
    for circle in (["ada", "ben", "cal"], ["dee", "eve", "fox"], ["gus", "hal", "ivy"]):
        for n in circle:
            g.add_node(n)
        for a, b in combinations(circle, 2):
            g.add_edge(a, b)
    g.add_node("jon")
    for contact in ("ada", "dee", "gus"):
        g.add_edge("jon", contact)
    return g


def main() -> int:
    g = build_circles()
    nb = Neighborhood(g)
    for who in ("jon", "ada", "ben"):
        print(nb.note(who))

    wc = WalkCount(g)
    print(f"walks: {wc.note(3)}; jon to eve in two steps: {wc.walks('jon', 'eve', 2)}")
    print(f"spectrum agrees at length four: {wc.matches_spectrum(4)}")

    by_hand = [["ada", "ben", "cal", "jon"], ["dee", "eve", "fox"], ["gus", "hal", "ivy"]]
    by_spectrum = SpectralClustering(g, 3).groups
    pc = PartitionCompare(g, by_hand, by_spectrum)
    print(f"hand against spectrum: {pc.note()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
