"""A shape day: rooms for meetings, a core with its fringe, and how surprising a graph is.

Run with: python -m examples.shapeday
"""

from __future__ import annotations

from mesh.cograph import Cograph
from mesh.entropy import GraphEntropy
from mesh.factories import complete, cycle, path, star
from mesh.graph import Graph
from mesh.intervalgraph import IntervalGraph
from mesh.splitgraph import SplitGraph
from mesh.walkkernel import WalkKernel


def build_core_and_fringe() -> Graph:
    g = Graph()
    core = ["ada", "ben", "cal", "dee"]
    for n in core:
        g.add_node(n)
    for i, a in enumerate(core):
        for b in core[i + 1 :]:
            g.add_edge(a, b)
    for fringe, contact in (("eve", "ada"), ("fox", "ada"), ("gus", "cal"), ("hal", "dee")):
        g.add_node(fringe)
        g.add_edge(fringe, contact)
    return g


def main() -> int:
    meetings = {
        "standup": (9.0, 9.5),
        "design": (9.25, 10.5),
        "review": (10.0, 11.0),
        "planning": (10.25, 10.75),
        "retro": (11.0, 12.0),
    }
    ig = IntervalGraph(meetings)
    print(f"meetings: {ig.note()}")
    rooms = ig.coloring()
    for name in sorted(rooms, key=lambda n: (rooms[n], n)):
        print(f"  room {rooms[name]}: {name}")

    g = build_core_and_fringe()
    sg = SplitGraph(g)
    print(f"office: {sg.note()}")
    print(f"  core {sorted(sg.clique)}, fringe {sorted(sg.fringe)}")

    cg = Cograph(g)
    print(f"office as a cograph: {cg.note()}")

    for name, shape in (("cycle", cycle(6)), ("star", star(5)), ("clique", complete(4))):
        print(f"{name:<7} {GraphEntropy(shape).note()}")

    kernel = WalkKernel(decay=0.1, depth=8)
    pairs = [("cycle-cycle", cycle(5), cycle(6)), ("cycle-star", cycle(5), star(5))]
    pairs.append(("path-clique", path(5), complete(5)))
    for label, a, b in pairs:
        print(f"{label:<12} similarity {kernel.similarity(a, b):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
