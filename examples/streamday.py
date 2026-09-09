"""A stream day: a contact log read at a moment, over a week, and along time itself.

Run with: python -m examples.streamday
"""

from __future__ import annotations

from mesh.graphbuilder import GraphBuilder
from mesh.graphstats import DegreeStats, power_law_graph
from mesh.linkstream import LinkStream


def contact_log() -> LinkStream:
    return LinkStream(
        [
            ("ada", "ben", 1, 2), ("ben", "cal", 3, 4), ("cal", "dee", 2, 3),
            ("dee", "eve", 5, 6), ("ada", "eve", 4, 5), ("eve", "fox", 7, 8),
            ("fox", "ada", 0, 1),
        ]
    )


def main() -> int:
    log = contact_log()
    print(f"snapshot at 2.5: {log.snapshot(2.5).edge_count()} live contact(s)")
    week = log.aggregate()
    print(f"aggregate: {week.edge_count()} pair(s) ever in contact")
    for source in ("ada", "fox"):
        print(log.note(source, 0))
        arrivals = log.reach(source, 0)
        earliest = ", ".join(f"{n}@{t:g}" for n, t in sorted(arrivals.items()))
        print(f"  earliest arrivals: {earliest}")

    built = GraphBuilder().text("ada ben\nben cal 2\ncal ada 3\n").node("dee")
    built.edge("dee", "ada")
    print(f"builder: {built.note()}")
    built.build()

    heavy = power_law_graph(400, gamma=2.3, kmin=2, seed=4)
    print(f"power-law graph: {DegreeStats(heavy).note(kmin=2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
