"""A matching day: pair workers with shifts three ways, then guard every edge.

Run with: python -m examples.matchingday
"""

from __future__ import annotations

from mesh.blossom import Blossom
from mesh.graph import Graph
from mesh.hopcroftkarp import HopcroftKarp
from mesh.hungarian import Hungarian
from mesh.vertexcover import VertexCover


def build_roster() -> tuple[list[str], list[str], list[tuple[str, str]]]:
    workers = ["ana", "bo", "cy", "dev"]
    shifts = ["dawn", "noon", "dusk", "night"]
    willing = [
        ("ana", "dawn"), ("ana", "noon"), ("bo", "noon"), ("bo", "dusk"),
        ("cy", "dusk"), ("cy", "night"), ("dev", "dawn"), ("dev", "night"),
    ]
    return workers, shifts, willing


def main() -> int:
    workers, shifts, willing = build_roster()
    hk = HopcroftKarp(workers, shifts)
    for w, s in willing:
        hk.add_edge(w, s)
    size = hk.solve()
    print(f"hopcroft-karp fills {size} of {len(shifts)} shifts: {hk.matching()}")

    # the same roster as a cost matrix: 1 where willing, 9 where not
    cost = [[1 if (w, s) in willing else 9 for s in shifts] for w in workers]
    hu = Hungarian(cost)
    assigned = {workers[i]: shifts[j] for i, j in enumerate(hu.assignment)}
    print(f"hungarian at total cost {hu.total}: {assigned}")

    g = Graph()
    for n in workers + shifts:
        g.add_node(n)
    for w, s in willing:
        g.add_edge(w, s)
    print(f"blossom on the same graph: {Blossom(g).size()} pairs, no odd cycles here")

    cover = VertexCover(g)
    print(f"vertex cover of {len(cover.cover)} guards every willing pair")
    print(f"  guards: {sorted(cover.cover)}")
    print(cover.note())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
