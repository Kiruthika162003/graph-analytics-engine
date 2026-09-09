"""A sample day: four ways to draw a piece of a graph, and what each one skews.

Run with: python -m examples.sampleday
"""

from __future__ import annotations

from mesh.factories import cycle
from mesh.graph import Graph
from mesh.graphsummary import GraphSummary
from mesh.sampling import Sampler


def build_town() -> Graph:
    g = cycle(16, prefix="r")
    g.add_node("square")
    for n in cycle(16, prefix="r").nodes():
        if int(n[1:]) % 2 == 0:
            g.add_edge("square", n)
    g.add_node("mill")
    for n in ("r1", "r3", "r5"):
        g.add_edge("mill", n)
    return g


def main() -> int:
    town = build_town()
    print(GraphSummary(town).lines()[0])
    s = Sampler(town, seed=11)
    print(s.note(s.by_nodes(6), "node"))
    print(s.note(s.by_edges(6), "edge"))
    print(s.note(s.snowball("square", 1), "snowball"))
    print(s.note(s.by_walk(12, start="r0"), "walk"))

    shares = s.visit_shares(5000, start="square")
    corrected = s.corrected_shares(5000, start="square")
    for name in ("square", "r7"):
        print(f"walk share of {name} {shares[name]:.3f}, corrected {corrected[name]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
