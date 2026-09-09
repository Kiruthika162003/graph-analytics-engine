"""A draw day: three layouts, one SVG, and a route explained edge by edge.

Run with: python -m examples.drawday
"""

from __future__ import annotations

from itertools import combinations

from mesh.graph import Graph
from mesh.layout import Layout
from mesh.pathexplain import PathExplanation
from mesh.spectralclustering import SpectralClustering
from mesh.svgrender import SvgRenderer


def build_village() -> Graph:
    g = Graph()
    for group in (["ada", "ben", "cal"], ["dee", "eve", "fox"]):
        for n in group:
            g.add_node(n)
        for a, b in combinations(group, 2):
            g.add_edge(a, b, 1.0)
    g.add_edge("cal", "dee", 3.0)
    g.add_edge("ben", "eve", 5.0)
    return g


def build_pipeline() -> Graph:
    g = Graph(directed=True)
    for n in ("fetch", "parse", "lint", "test", "build", "ship"):
        g.add_node(n)
    arcs = [
        ("fetch", "parse"), ("parse", "lint"), ("parse", "test"), ("lint", "build"),
        ("test", "build"), ("build", "ship"),
    ]
    for u, v in arcs:
        g.add_edge(u, v)
    return g


def main() -> int:
    village = build_village()
    lay = Layout(village)
    for method, pos in (("circle", lay.circle()), ("spring", lay.spring(rounds=150, seed=1))):
        print(lay.note(pos, method))

    groups = SpectralClustering(village, 2).groups
    palette = ("#f4a", "#4af")
    renderer = SvgRenderer(village, lay.spring(rounds=150, seed=1))
    for color, group in zip(palette, groups, strict=False):
        for node in group:
            renderer.color(node, color)
    svg = renderer.render()
    print(f"{renderer.note()}; {svg.count('<circle')} circle(s), {svg.count('<line')} line(s)")

    pipeline = build_pipeline()
    layered = Layout(pipeline).layered()
    depth = len({round(y, 3) for _x, y in layered.values()})
    print(f"pipeline laid out in {depth} layer(s); ship sits at y={layered['ship'][1]:.1f}")

    pe = PathExplanation(village, "ada", "fox")
    print(pe.note())
    for a, b, w, running in pe.steps():
        print(f"  {a} > {b} costs {w:g}, {running:g} so far, slack {pe.slack()[(a, b)]:g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
