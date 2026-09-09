"""A logic day: a roster as 2-SAT, a pipeline with minimums, two teams split, motif distances.

Run with: python -m examples.logicday
"""

from __future__ import annotations

from mesh.circulation import Circulation
from mesh.editdistance import EditDistance
from mesh.factories import cycle, path, star
from mesh.graph import Graph
from mesh.kernighanlin import KernighanLin
from mesh.treehash import TreeHash
from mesh.twosat import Clause, TwoSat


def roster_clauses() -> list[Clause]:
    # ada or ben covers monday; if ada works monday she is off tuesday; ben and cal
    # cannot both be off tuesday; cal works tuesday only if ada does not work monday
    return [
        (("ada_mon", True), ("ben_mon", True)),
        (("ada_mon", False), ("ada_tue", False)),
        (("ben_tue", True), ("cal_tue", True)),
        (("cal_tue", False), ("ada_mon", False)),
    ]


def build_pipeline() -> Graph:
    g = Graph(directed=True)
    for n in ("well", "plant", "town", "farm"):
        g.add_node(n)
    for u, v, cap in [("well", "plant", 9), ("plant", "town", 6), ("plant", "farm", 5)]:
        g.add_edge(u, v, cap)
    g.add_edge("town", "well", 7)
    g.add_edge("farm", "well", 4)
    return g


def build_two_teams() -> Graph:
    g = Graph()
    left = ["ada", "ben", "cal", "dee"]
    right = ["eve", "fox", "gus", "hal"]
    for team in (left, right):
        for n in team:
            g.add_node(n)
        for i, a in enumerate(team):
            for b in team[i + 1 :]:
                g.add_edge(a, b)
    g.add_edge("dee", "eve")
    return g


def main() -> int:
    ts = TwoSat(roster_clauses())
    print(f"roster: {ts.note()}")

    pipe = Circulation(build_pipeline(), {("plant", "town"): 4, ("plant", "farm"): 3})
    print(f"pipeline: {pipe.note()}")
    for (u, v), f in sorted(pipe.circulation.items()):
        print(f"  {u:>5} -> {v:<5} carries {f:g}")

    kl = KernighanLin(build_two_teams(), left={"ada", "ben", "eve", "fox"})
    print(f"teams: {kl.note()}")

    print(f"path-3 vs triangle: {EditDistance(path(3), cycle(3)).note()}")
    print(f"star-3 vs path-4: {EditDistance(star(3), path(4)).note()}")
    print(f"tree names: path {TreeHash(path(4)).canonical}, star {TreeHash(star(3)).canonical}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
