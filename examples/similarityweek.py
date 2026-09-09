"""A similarity week: common shapes, graphlet peers, walk kernels, and edit distance together.

Run with: python -m examples.similarityweek
"""

from __future__ import annotations

from mesh.commonsubgraph import CommonSubgraph
from mesh.editdistance import EditDistance
from mesh.factories import cycle, path, star, wheel
from mesh.graph import Graph
from mesh.graphlets import Graphlets
from mesh.walkkernel import WalkKernel


def build_office() -> Graph:
    g = Graph()
    ties = [
        ("ada", "ben"), ("ada", "cal"), ("ben", "cal"), ("cal", "dee"), ("dee", "eve"),
        ("dee", "fox"), ("eve", "fox"), ("fox", "gus"), ("gus", "hal"),
    ]
    for a, b in ties:
        g.add_node(a)
        g.add_node(b)
        g.add_edge(a, b)
    return g


def main() -> int:
    shapes = {"path": path(5), "cycle": cycle(5), "star": star(4), "wheel": wheel(4)}
    kernel = WalkKernel(decay=0.1, depth=8)
    print("pairwise readings: common nodes / edit distance / walk similarity")
    names = list(shapes)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            common = CommonSubgraph(shapes[a], shapes[b]).size()
            edits = EditDistance(shapes[a], shapes[b]).distance
            walk = kernel.similarity(shapes[a], shapes[b])
            print(f"  {a:<5} vs {b:<5} common {common}, edits {edits}, walk {walk:.3f}")

    office = build_office()
    gl = Graphlets(office)
    print(gl.note("dee"))
    peers = ", ".join(f"{n} ({d:.2f})" for n, d in gl.peers("dee")[:3])
    print(f"dee's closest structural peers: {peers}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
